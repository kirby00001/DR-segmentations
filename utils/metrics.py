import torch
import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def pr_auc(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if y_true.max() != 0:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
    else:
        pr_auc = -1
    return pr_auc


def mauc_coef(y_true, y_pred):
    y_true = y_true.int().cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    auc_list = []
    for i in range(1, 5):
        auc_list.append(pr_auc(y_true[:, i, :, :], y_pred[:, i, :, :]))
    masked_auc_list = np.ma.masked_equal(auc_list, -1)
    print(masked_auc_list)
    mauc = np.mean(masked_auc_list)
    return auc_list[0], auc_list[1], auc_list[2], auc_list[3], mauc


def dice_coef(y_true, y_pred, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


if __name__ == "__main__":
    y_true = torch.tensor(
        [
            [
                [[1, 0, 0, 0, 0]],
                [[0, 1, 0, 0, 0]],
                [[0, 0, 1, 0, 0]],
                [[0, 0, 0, 1, 1]],
                [[0, 0, 0, 0, 0]],
            ]
        ]
    )
    y_pred = torch.tensor(
        [
            [
                [[0.6, 0.1, 0.1, 0.1, 0.1]],
                [[0.1, 0.6, 0.1, 0.1, 0.1]],
                [[0.1, 0.1, 0.6, 0.1, 0.1]],
                [[0.1, 0.1, 0.1, 0.6, 0.6]],
                [[0.1, 0.1, 0.1, 0.1, 0.1]],
            ]
        ]
    )
    print(mauc_coef(y_true, y_pred))
    print(dice_coef(y_true, y_pred))
    print(iou_coef(y_true, y_pred))
