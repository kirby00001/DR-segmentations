import pickle

from torchinfo import summary
import segmentation_models_pytorch as smp

from colorama import Fore,Style

from train import valid_one_epoch
from datasets.IDRiD import get_dataloader_IDRiD
from utils import get_transform
from .improvedunet import improvedunet

def unet(
    classes=5,
    in_channels=3,
):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.Unet(
        classes=classes,
        in_channels=in_channels,
    )


def unetplusplus(
    classes=5,
    in_channels=3,
):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.UnetPlusPlus(
        classes=classes,
        in_channels=in_channels,
    )


def pspnet(
    classes=5,
    in_channels=3,
):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.PSPNet(
        classes=classes,
        in_channels=in_channels,
    )


def deeplabv3plus(
    classes=5,
    in_channels=3,
):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    return smp.DeepLabV3Plus(
        classes=classes,
        in_channels=in_channels,
    )

def evaluate(model, loss_fn, device, config):
    args = pickle.load(open("./configs/arg.bin",'rb'))
    color = Fore.GREEN
    reset = Style.RESET_ALL
    # Load Data
    valid_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="valid"),
        shuffle=True,
        mode="valid",
    )
    # validation
    _, val_scores = valid_one_epoch(
        model=model,
        dataloader=valid_dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    model=config["model"]["model"]
    val_scores=args[model]
    ex_auc, he_auc, ma_auc, se_auc, mean_auc, dice, iou = val_scores
    print(
        f"{color}EX AUC: {ex_auc:0.4f} | HE AUC: {he_auc:0.4f} | MA AUC: {ma_auc:0.4f} | SE AUC: {se_auc:0.4f}"
    )
    print(f"Mean AUC: {mean_auc:0.4f} | {color}Dice: {dice:0.4f} | IoU: {iou:0.4f}{reset}")
    
def get_model(model_name):
    if model_name == "improvedunet":
        model = improvedunet()
    elif model_name == "unet":
        model = unet()
    elif model_name == "unet++":
        model = unetplusplus()
    elif model_name == "pspnet":
        model = pspnet()
    elif model_name == "deeplabv3+":
        model = deeplabv3plus()
    else:
        raise
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_depth=4,
        decoder_channels=[32, 96, 144, 240],
        in_channels=3,
        classes=5,
        decoder_use_batchnorm=True,
        activation=None,
    )
    
if __name__ == "__main__":
    # model = unet()
    # model = unetplusplus()
    # model = pspnet()
    model = deeplabv3plus()
    # model summary
    summary(model, input_size=(1, 3, 960, 1440), device="cpu", depth=3)
