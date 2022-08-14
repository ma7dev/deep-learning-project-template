from typing import Optional

import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class LitDataset(pl.LightningDataModule):
    def __init__(
        self: pl.LightningDataModule,
        data_path: str,
        batch_size: int,
        num_workers: int = 4,
        num_classes: int = 10,
        transform_fn: T.Compose = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.transform_fn = transform_fn
        self.split = [55000, 5000]

    def prepare_data(self: pl.LightningDataModule) -> None:
        MNIST(self.data_path, train=True, download=True)
        MNIST(self.data_path, train=False, download=True)

    def setup(
        self: pl.LightningDataModule, stage: Optional[str] = None
    ) -> None:
        if stage == "fit" or stage is None:
            dataset = MNIST(
                self.data_path,
                train=True,
                transform=self.transform_fn(is_train=True),
            )
            (self.train_dataset, self.val_dataset) = random_split(
                dataset, self.split
            )

        else:
            self.test_dataset = MNIST(
                self.data_path, train=False, transform=self.transform_fn()
            )

    def train_dataloader(self: pl.LightningDataModule) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self: pl.LightningDataModule) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self: pl.LightningDataModule) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
