import yaml
import argparse

import torch

from utils import get_loss
from models import evaluate, get_model



if __name__ == "__main__":
    # 解析配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--weight")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    best_weight = torch.load(args.weight)
    # print(config)
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
    evaluate(model, loss_fn = get_loss(loss), device = "cuda", config = config)
