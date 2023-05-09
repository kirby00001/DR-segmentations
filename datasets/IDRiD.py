from glob import glob
from torch.utils.data import Dataset, DataLoader
from utils import get_transform, load_image, load_mask, image_show


class IDRiD(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_image(img_path)
        mask = load_mask(img_path)

        data = self.transform(image=img, mask=mask)
        img, mask = data["image"], data["mask"]

        return img, mask


def get_dataloader_IDRiD(transform, batch_size, shuffle, mode):
    if mode == "train":
        img_paths = glob(
            "./data/IDRiD/A. Segmentation/1. Original Images/a. Training Set/*.jpg"
        )
    elif mode == "valid":
        img_paths = glob(
            "./data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set/*.jpg"
        )
    else:
        raise ValueError("mode = 'train'|'valid'")
    img_paths.sort()
    dataset = IDRiD(img_paths=img_paths, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


if __name__ == "__main__":
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

    image, masks = next(iter(train_dataloader))
    # image, masks = next(iter(test_dataloader))
    print("img.max:", image.max())
    print("img.min:", image.min())
    print("img.shape:", image.shape)
    print("mask.max:", masks.max())
    print("mask.min", masks.min())
    print("mask.shape:", masks.shape)
