from typing import Optional, Union, List

import torch
from torch import nn
import torchinfo

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead


def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MixAtt(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpoll = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels, in_channels, (1, 1), stride=1, padding="same"
        )
        self.dense = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, concat):
        # Channel Attentation
        shape = concat.shape
        x = self.avgpoll(concat).view(shape[0], shape[1])
        x = self.dense(x).view(shape[0], shape[1], 1, 1)
        score_c = x
        CA = concat * score_c.expand_as(concat)

        # Spatial Attentaion
        s_avg = torch.mean(concat, dim=1, keepdim=True)
        score_s = self.sigmoid(s_avg)
        SA = concat * score_s

        # Point wise Attention
        x = concat
        score_p = self.sigmoid(x)
        PA = concat * score_p

        out = CA + SA + PA
        return out


class DenseCat(nn.Module):
    def __init__(self, level, channels_size=[32, 24, 40, 112]) -> None:
        super().__init__()
        self.level = level
        self.dab = MixAtt(channels_size[level])
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels_size[i],
                    channels_size[level] // 4,
                    (1, 1),
                    stride=1,
                    padding="same",
                )
                for i in range(4)
            ]
        )

    def forward(self, features):
        current = features[self.level]
        shape = current.shape
        tensors = []
        for i in range(4):
            if i < self.level:
                tensor = nn.AdaptiveMaxPool2d((shape[2], shape[3]))(features[i])
            elif i > self.level:
                tensor = nn.Upsample(size=(shape[2], shape[3]), mode="bilinear")(
                    features[i]
                )
            else:
                tensor = features[i]
            tensor = self.convs[i](tensor)
            tensors.append(tensor)

        concat = torch.cat(tensors, dim=1)
        x = self.dab(concat)
        return x


class DMUnet(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,  # type: ignore
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.pffs = nn.ModuleList([DenseCat(level=i) for i in range(4)])

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        for pff in self.pffs:
            initialize_decoder(pff)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        for i in range(4):
            features[i + 1] = self.pffs[i](features[1:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


if __name__ == "__main__":
    model = DMUnet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=5,
        encoder_depth=4,
        decoder_channels=[240, 144, 96, 32],
        in_channels=3,
        decoder_use_batchnorm=True,
        activation=None,
    )
    torchinfo.summary(model, input_size=(1, 3, 960, 1440), device="cpu", depth=3)
