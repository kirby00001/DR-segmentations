import torch
from torch.nn import CrossEntropyLoss
import segmentation_models_pytorch as smp



def get_loss(mode):
    if mode=="CrossEntropy Loss":
        return CrossEntropyLoss(
        reduction="mean",
    )
    elif mode=="Focal Loss":
        return smp.losses.FocalLoss(
        mode="multilabel"
    )
    else:
        raise



if __name__ == "__main__":
    loss_fn = get_loss("CrossEntropyLoss")
    input = torch.rand(size=(1, 5, 480, 720))
    target = torch.randint(low=0, high=1, size=(1, 5, 480, 720))
    print(loss_fn(input, target))
