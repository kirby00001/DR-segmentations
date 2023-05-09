import gc
import copy
import time
import wandb
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn.functional as F

from datasets.IDRiD import get_dataloader_IDRiD
from models.DMUnet import DMUnet
from utils import get_transform, get_loss, mauc_coef, dice_coef, iou_coef

from colorama import Fore, Style
from collections import defaultdict

color = Fore.GREEN
reset = Style.RESET_ALL


def train_one_epoch(model, optimizer, loss_fn, dataloader, device):
    model.train()
    model.to(device)

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    for _, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        batch_size = images.shape[0]

        y_pred = model(images)
        loss = loss_fn(y_pred, masks)

        # back propgation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate running loss and epoch_loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        # display info
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            train_epoch_loss=f"{epoch_loss:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )

    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    model.to(device)

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0
    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")
    for _, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        batch_size = images.shape[0]

        y_pred = model(images)
        loss = loss_fn(y_pred, masks)

        # calculate running loss and epoch_loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        y_score = F.softmax(y_pred, dim=1)
        ma_auc, he_auc, ex_auc, se_auc, mean_auc = mauc_coef(
            y_true=masks, y_pred=y_score
        )
        dice = dice_coef(y_true=masks, y_pred=y_score).cpu().detach().numpy()
        iou = iou_coef(y_true=masks, y_pred=y_score).cpu().detach().numpy()
        val_scores.append([ma_auc, he_auc, ex_auc, se_auc, mean_auc, dice, iou])

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            valid_epoch_loss=f"{epoch_loss:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )
    masked_val_scores = np.ma.masked_equal(val_scores, -1)
    val_scores = np.mean(masked_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores


def run_training(model, loss_fn, optimizer, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    # Initilization
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_iou = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    # Load Data
    train_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="train"),
        shuffle=True,
        mode="train",
    )
    valid_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="valid"),
        shuffle=True,
        mode="valid",
    )

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            device=device,
        )

        val_loss, val_scores = valid_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        ma_auc, he_auc, ex_auc, se_auc, mean_auc, dice, iou = val_scores
        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["MA PR AUC"].append(ma_auc)
        history["HE PR AUC"].append(he_auc)
        history["EX PR AUC"].append(ex_auc)
        history["SE PR AUC"].append(se_auc)
        history["Mean PR AUC"].append(mean_auc)
        history["Valid Dice"].append(dice)
        history["Valid IoU"].append(iou)

        # Log loss and metrics
        wandb.log(
            {
                "Train Loss": train_loss,
                "Valid Loss": val_loss,
                "MA PR AUC": ma_auc,
                "HE PR AUC": he_auc,
                "EX PR AUC": ex_auc,
                "SE PR AUC": se_auc,
                "Mean PR AUC": mean_auc,
                "Valid Dice": dice,
                "Valid IoU": iou,
                # "LR": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"MA AUC: {ma_auc:0.4f} | HE AUC: {he_auc:0.4f} | EX AUC: {ex_auc:0.4f} | SE AUC: {se_auc:0.4f} | Mean AUC: {mean_auc:0.4f}"
        )
        print(f"Dice: {dice:0.4f} | IoU: {iou:0.4f}")

        # deep copy the model
        if dice > best_dice:
            print(f"{color}Valid Score Improved ({best_dice:0.4f} ---> {dice:0.4f})")
            best_mean_auc = mean_auc
            best_dice = dice
            best_iou = iou
            best_epoch = epoch
            if run != None:
                run.summary["Best Mean AUC"] = best_mean_auc
                run.summary["Best Dice"] = best_dice
                run.summary["Best IoU"] = best_iou
                run.summary["Best Epoch"] = best_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"./checkpoints/best_epoch.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{reset}")

        last_model_weights = copy.deepcopy(model.state_dict())
        PATH = f"./checkpoints/last_epoch.bin"
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Score: {:.4f}".format(best_iou))
    # load best model weights
    model.load_state_dict(best_model_weights)

    if run != None:
        run.finish()

    return model, history


if __name__ == "__main__":

    wandb_key = "b9b9bfc9d98eada98a991a294a1e40ad81437726"
    anonymous = None

    encoder_name = "efficientnet-b0"
    encoder_weights = "imagenet"
    num_class = 5

    epoch = 10
    device = "cpu"
    loss = "Focal Loss"

    wandb.login(key=wandb_key)

    run = wandb.init(
        project="DR Segmentation",
        name=f"Dim 960x1440|model DMUNet+",
        anonymous=anonymous,
        group=f"DMUnet {encoder_name} 960x1440",
        config={
            "device": device,
            "epoch": epoch,
            "loss": loss,
        },
    )

    model = DMUnet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_class,
        encoder_depth=4,
        decoder_channels=[240, 144, 96, 32],
        in_channels=3,
        decoder_use_batchnorm=True,
        activation=None,
    )

    run_training(
        model=model,
        loss_fn=get_loss(loss),
        optimizer=Adam(model.parameters(), lr=2e-3),
        device=device,
        num_epochs=epoch,
    )
