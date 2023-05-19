import yaml
import argparse

import wandb
from torchinfo import summary

from utils import get_loss
from torch.optim import Adam
from models import unet, unetplusplus, pspnet, deeplabv3plus, improvedunet

from train import run_training

# 解析配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()
config = yaml.safe_load(open(args.config, "r"))

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
if model_name == "improvedunet":
    model = improvedunet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_class,
    )
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

summary(model, input_size=(1, 3, 960, 1440), device="cuda", depth=3)

# 根据配置初始化Wandb
wandb.login(key=wandb_key)
run = wandb.init(
    project="DR Segmentation",
    name=f"Dim 960x1440 | model {model_name}",
    anonymous=anonymous,
    group=f"{model_name} {encoder_name} 960x1440",
    config={
        "epoch": epoch,
        "loss": loss,
    },
)

# 根据配置训练模型
run_training(
    model=model,
    loss_fn=get_loss(loss),
    optimizer=Adam(model.parameters(), lr=2e-3),
    device=device,
    num_epochs=epoch,
    run=run,
)
