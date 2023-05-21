import time

import yaml
import argparse

import torch

from utils import get_transform, get_model, get_loss
from train import valid_one_epoch
from datasets.IDRiD import get_dataloader_IDRiD


def evaluate(model, loss_fn, device):
    # Load Data
    valid_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="valid"),
        shuffle=True,
        mode="valid",
    )
    # validation
    val_loss, val_scores = valid_one_epoch(
        model=model,
        dataloader=valid_dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    end = time.time()
    # output
    ma_auc, he_auc, ex_auc, se_auc, mean_auc, dice, iou = val_scores
    print(
        f"MA AUC: {ma_auc:0.4f} | HE AUC: {he_auc:0.4f} | EX AUC: {ex_auc:0.4f} | SE AUC: {se_auc:0.4f} | Mean AUC: {mean_auc:0.4f}"
    )
    print(f"Dice: {dice:0.4f} | IoU: {iou:0.4f}")


if __name__ == "__main__":
    # 解析配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--weight")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    best_weight = torch.load(args.weight)

    print(config)
    # wandb
    wandb_key = config["wandb"]["wandb_key"]
    anonymous = None
    # model
    model_name = config["model"]["model"]
    encoder_name = config["model"]["encoder_name"]
    encoder_weights = config["model"]["encoder_weights"]
    num_class = config["model"]["num_class"]
    # train
    epoch = config["train"]["epoch"]
    device = config["train"]["device"]
    loss = config["train"]["loss"]

    # 根据配置文件获得对应模型
    model = get_model(model_name)
    # 加载模型最有权重
    model.load_state_dict(best_weight)
    # 评估模型
    evaluate(model, loss_fn=get_loss(loss), device="cpu")
