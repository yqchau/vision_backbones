"""
Purpose: Setting up dataset and dataloader.

Code is set up in the format of Dataset class follow by all related DataModule/DataLoader class that uses the Dataset.
"""

import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from hydra.utils import instantiate
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from .utils import make_weights_for_balanced_classes


class CutMixCollator:
    def __init__(self, alpha=1, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def cutmix(self, batch):

        data, targets, path = batch
        probability = np.random.rand(1)

        if self.prob > probability:

            lam = np.random.beta(self.alpha, self.alpha)
            image_h, image_w = data.shape[2:]
            cx = np.random.uniform(0, image_w)
            cy = np.random.uniform(0, image_h)
            w = image_w * np.sqrt(1 - lam)
            h = image_h * np.sqrt(1 - lam)
            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, image_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, image_h)))

            indices = torch.randperm(data.size(0))
            data[:, :, y0:y1, x0:x1] = data[indices][
                :, :, y0:y1, x0:x1
            ]  # shuffle data and overlay on original data
            targets = (targets, targets[indices], lam)

        return data, targets, path

    def __call__(self, batch):
        batch = default_collate(batch)  # collate in batch form as per usual
        batch = self.cutmix(batch)
        return batch


class AlbuImageFolder(torchvision.datasets.ImageFolder):
    """Re-use ImageFolder that was coded for torchvision library for the use of
    Albumentation library. Do not need to make own custom dataset.

    Link to torchvision.datasets.ImageFolder source code:

    https://github.com/pytorch/vision/blob/d0dede0e09d6d72253080cb366742a270f0ea8cd/torchvision/datasets/folder.py#L108

    Difference is that Albumentation takes in numpy array while torchvision loads pillow images. Transform for torchvision expects Pillow,
    while Albumentation expects numpy array (opencv). Thus, we only need to override __getitem__.
    """

    def __init__(self, path, transform, grayscale_transform=None):
        super().__init__(root=path, transform=transform)
        self.grayscale_transform = grayscale_transform

    def cv2_loader(self, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int):
        """Overiding function to load and transform following Albumentation
        format.

        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.cv2_loader(path)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

            if self.grayscale_transform is not None:
                image = self.grayscale_transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, path


class AlbuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        workers: int,
        num_classes: int,
        image_width: int,
        image_height: int,
        train_transform: dict,
        test_transform: dict,
        alpha_cutmix: float = 1.0,
        prob_cutmix: float = 0.5,
        mean=[0.485, 0.456, 0.406],
        std_dev=[0.229, 0.224, 0.225],
        weighted_sampling: bool = False,
        in_chans: int = 3,
    ):
        super().__init__()
        self.data_path = data_path
        self.prob_cutmix = prob_cutmix
        self.alpha_cutmix = alpha_cutmix
        self.batch_size = batch_size
        self.workers = workers
        self.num_classes = num_classes
        self.img_width = image_width
        self.img_height = image_height
        self.mean = mean
        self.std_dev = std_dev
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.weighted_sampling = weighted_sampling
        self.in_chans = in_chans

    def setup(self, stage=None):
        # In setup, we will set up the data flow (dataset+data aug) for the Dataloader.
        # Lightning will auto call this function and `prepare_dataset`
        train_transform = instantiate(self.train_transform, _convert_="partial")
        test_transform = instantiate(self.test_transform, _convert_="partial")
        grayscale_transform = transforms.Grayscale(num_output_channels=1)

        train_dir = os.path.join(self.data_path, "train")
        val_dir = os.path.join(self.data_path, "val")

        if stage in (None, "fit"):
            self.train_dataset = AlbuImageFolder(
                train_dir,
                train_transform,
                grayscale_transform if self.in_chans == 1 else None,
            )

            self.eval_dataset = AlbuImageFolder(
                val_dir,
                test_transform,
                grayscale_transform if self.in_chans == 1 else None,
            )

        # DATA Sanity Check:
        train_class = self.train_dataset.classes  # class labels
        val_class = self.eval_dataset.classes

        if self.weighted_sampling:
            weights = make_weights_for_balanced_classes(
                self.train_dataset.imgs, len(self.train_dataset.classes)
            )
            weights = torch.DoubleTensor(weights)
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, len(weights)
            )

        if train_class != val_class:
            raise SystemExit(
                "Training class labels and validation class labels is different"
            )

        if len(train_class) != self.num_classes:
            raise SystemExit(
                "Number of classes in training dataset differs from the number of classes set in configuration"
            )

    # Visualise Data
    def visualize_augmentations(
        self, which_dataset="train", idx=0, samples=100, cols=10
    ):

        # remember to change to hard copy for the dataset especially if visualisation function runs before/during any training/testing related code.
        dataset = self.train_dataset if which_dataset == "train" else self.eval_dataset

        dataset.transform = A.Compose(
            dataset.transform[:-2]
        )  # ignore normalising and conversion to pytorch tensor.
        rows = samples // cols
        _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(rows, 6))
        for i in range(samples):
            image, _ = dataset[idx + i]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    # Set up the DataLoaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if self.weighted_sampling else True,
            sampler=self.sampler if self.weighted_sampling else None,
            num_workers=self.workers,
            collate_fn=CutMixCollator(alpha=self.alpha_cutmix, prob=self.prob_cutmix)
            if self.prob_cutmix > 0.0
            else default_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":

    path = "/home/ml2/imagewoof2-small"
    batch_size = 32
    num_workers = 4

    train_transform = {
        "_target_": "albumentations.Compose",
        "transforms": [
            {
                "_target_": "albumentations.Resize",
                "height": 256,
                "width": 256,
                "interpolation": 2,
            },
            {
                "_target_": "albumentations.RandomCrop",
                "height": 224,
                "width": 224,
            },
            {
                "_target_": "albumentations.Normalize",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            {"_target_": "albumentations.pytorch.ToTensorV2"},
        ],
    }

    data = AlbuDataModule(
        data_path=path,
        batch_size=batch_size,
        workers=num_workers,
        num_classes=10,
        prob_cutmix=0.0,
        alpha_cutmix=1,
        image_height=224,
        image_width=224,
        train_transform=train_transform,
        test_transform=None,
    )

    data.setup()
    # Visualise transforms:
    # data.visualize_augmentations(idx=2000)

    loader = data.train_dataloader()
    # Visualise loader outputs
    for img, labels, _ in loader:
        grid_img = torchvision.utils.make_grid(img, nrow=10)
        break
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
