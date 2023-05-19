from torchinfo import summary
import segmentation_models_pytorch as smp

# TODO: Unet
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


# TODO: Unet ++
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


# TODO: PSPNet
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


# TODO: DeepLabv3+
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


if __name__ == "__main__":
    model = unetplusplus()
    # model summary
    summary(model, input_size=(1, 3, 960, 1440), device="cpu", depth=5)
