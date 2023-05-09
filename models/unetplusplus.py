from torchinfo import summary
import segmentation_models_pytorch as smp


def get_model_unetplusplus(
    encoder_name,
    encoder_weights,
    classes,
    encoder_depth=4,
    decoder_channels=[32, 96, 144, 240],
    in_channels=3,
    decoder_use_batchnorm=True,
    activation=None,
):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        classes=classes,
        in_channels=in_channels,
        decoder_use_batchnorm=decoder_use_batchnorm,  # type: ignore
        activation=activation,
    )


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=5,
    )
    # model summary
    summary(model, input_size=(1, 3, 960, 1440), device="cuda", depth=5)
