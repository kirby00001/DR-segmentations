import cv2
import glob
import torch
import numpy as np
from PIL import Image


def load_image(image_path):
    img = np.array(Image.open(image_path))
    return img/255.0


# data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms
def path2paths(image_path):
    infos = image_path.split("/")
    train_or_test = infos[-2]
    # print("train_or_test:", train_or_test)
    i = infos[-1][:-4]
    # print("ID:",i)
    return glob.glob(
        f"./data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/{train_or_test}/[1234]*/{i}_*.tif"
    )


def path2id(mask_path):
    return int(mask_path.split("/")[-2][0])


def mask2label(masks):
    pad_masks = np.pad(masks, pad_width=[(0, 0), (0, 0), (1, 0)])
    label = np.argmax(pad_masks, axis=-1)
    return label


def load_mask(image_path):
    mask_paths = path2paths(image_path)
    masks = np.zeros(shape=(2848, 4288, 4))
    for path in mask_paths:
        # print("path2id:",path2id(path))
        image_id = path2id(path)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        masks[:, :, image_id - 1] = mask
    label = torch.Tensor(mask2label(masks=masks)).long()
    return torch.nn.functional.one_hot(label, num_classes=5).detach().numpy()


if __name__ == "__main__":
    # img_path = "./data/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_17.jpg"
    iamge_path = "./data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_81.jpg"
    # load image
    image = load_image(iamge_path)
    print("img.max:", image.max())
    print("image.shape", image.shape)
    # load masks
    masks = load_mask(iamge_path)
    print("mask.max", masks.max())
    print("mask.shape", masks.shape)
