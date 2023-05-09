import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = {
    "train": A.Compose(
        [
            # A.RandomResizedCrop(960, 1440, scale=(2000, 2000)),
            A.Resize(960, 1440),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(transpose_mask=True),
        ]
    ),
    "valid": A.Compose(
        [
            A.Resize(960, 1440),
            ToTensorV2(transpose_mask=True),
        ]
    ),
}


def get_transform(mode):
    return transforms[mode]
