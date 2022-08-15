from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import cool_project.optims as optims
from torch.optim import Adam
from torchmetrics import Accuracy, F1Score, Precision, Recall

from cool_project.models.baseline import Baseline


class LitModel(pl.LightningModule):
    def __init__(
        self: pl.LightningModule,
        model_config: Any,
        optimizer_config: Any,
        lr_config: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.lr_config = lr_config
        self.num_classes = self.model_config.num_classes
        self.model = Baseline(**self.model_config.dict())

        self.criterion = F.nll_loss
        self.metrics = {
            "train": {
                "acc": Accuracy(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
                "prec": Precision(
                    num_classes=self.num_classes, average="macro"
                ),
                "rec": Recall(num_classes=self.num_classes, average="macro"),
            },
            "val": {
                "acc": Accuracy(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
                "prec": Precision(
                    num_classes=self.num_classes, average="macro"
                ),
                "rec": Recall(num_classes=self.num_classes, average="macro"),
            },
            "test": {
                "acc": Accuracy(num_classes=self.num_classes, average="macro"),
                "f1": F1Score(num_classes=self.num_classes, average="macro"),
                "prec": Precision(
                    num_classes=self.num_classes, average="macro"
                ),
                "rec": Recall(num_classes=self.num_classes, average="macro"),
            },
        }

        self.automatic_optimization = False

    def configure_optimizers(
        self: pl.LightningModule,
    ) -> torch.optim.Optimizer:
        optimizer = Adam(self.parameters(), **self.optimizer_config.kwargs)
        return optimizer

    def forward(self: pl.LightningModule, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # logging
    def logging_step(
        self: pl.LightningModule,
        loss: Dict[Any, Any],
        acc: Dict[Any, Any],
        mode: str,
    ) -> None:
        self.log(f"{mode}/step/loss", loss, on_step=True, rank_zero_only=True)
        for metric_name, ac in acc.items():
            self.log(
                f"{mode}/step/{metric_name}",
                ac,
                on_step=True,
                rank_zero_only=True,
            )

    def logging_epoch(
        self: pl.LightningModule, epoch_outputs: Dict[Any, Any], mode: str
    ) -> None:

        loss = torch.stack([x["loss"].detach() for x in epoch_outputs]).mean()

        self.log(
            f"{mode}/epoch/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        for metric_name in self.metrics[mode].keys():
            acc = self.metrics[mode][metric_name].compute()
            self.log(
                f"{mode}/epoch/{metric_name}",
                acc,
                on_epoch=True,
                prog_bar=True,
                rank_zero_only=True,
            )
            self.metrics[mode][metric_name].reset()

    # compute
    def compute_metrics(
        self: pl.LightningModule,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mode: str,
    ) -> Dict[Any, Any]:
        preds = preds.detach().cpu().int()
        targets = targets.detach().cpu().int()

        acc = {}
        for metric_name in self.metrics[mode].keys():
            acc[metric_name] = self.metrics[mode][metric_name](preds, targets)
        return acc

    def _step(
        self: pl.LightningModule, batch: List[Any], batch_idx: int, mode: str
    ) -> Dict[Any, Any]:
        if mode == "train":
            optimizer = self.optimizers()
            images, targets = batch
            outputs = self(images)
            loss = self.criterion(outputs, targets)
            preds = torch.argmax(outputs, dim=1)

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

        else:
            images, targets = batch
            outputs = 0
            loss = 0
            with torch.no_grad():
                outputs = self(images)
                loss = self.criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

        return {
            "loss": loss.clone(),
            "preds": preds.clone().detach(),
            "targets": targets.clone().detach(),
        }

    def training_step(
        self: pl.LightningModule, batch: List[Any], batch_idx: int
    ) -> Dict[Any, Any]:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self: pl.LightningModule, batch: List[Any], batch_idx: int
    ) -> Dict[Any, Any]:
        return self._step(batch, batch_idx, "val")

    def test_step(
        self: pl.LightningModule, batch: List[Any], batch_idx: int
    ) -> Dict[Any, Any]:
        return self._step(batch, batch_idx, "test")

    def _step_end(
        self: pl.LightningModule, step_outputs: Dict[Any, Any], mode: str
    ) -> None:
        loss = step_outputs["loss"].detach()
        acc = self.compute_metrics(
            step_outputs["preds"], step_outputs["targets"], mode
        )
        self.logging_step(loss, acc, mode)

    def training_step_end(
        self: pl.LightningModule, step_outputs: Dict[Any, Any]
    ) -> None:
        self._step_end(step_outputs, "train")

    def validation_step_end(
        self: pl.LightningModule, step_outputs: Dict[Any, Any]
    ) -> None:
        self._step_end(step_outputs, "val")

    def test_step_end(
        self: pl.LightningModule, step_outputs: Dict[Any, Any]
    ) -> None:
        self._step_end(step_outputs, "test")

    # _epoch_end
    def _epoch_end(
        self: pl.LightningModule, epoch_outputs: Dict[Any, Any], mode: str
    ) -> None:
        # if mode == 'train':
        #     lr_scheduler = self.lr_schedulers()
        #     lr_scheduler.step()
        self.logging_epoch(epoch_outputs, mode)

    def training_epoch_end(
        self: pl.LightningModule, epoch_outputs: Dict[Any, Any]
    ) -> None:
        self._epoch_end(epoch_outputs, "train")

    def validation_epoch_end(
        self: pl.LightningModule, epoch_outputs: Dict[Any, Any]
    ) -> None:
        self._epoch_end(epoch_outputs, "val")

    def test_epoch_end(
        self: pl.LightningModule, epoch_outputs: Dict[Any, Any]
    ) -> None:
        self._epoch_end(epoch_outputs, "test")

    def predict_step(
        self: pl.LightningModule, images: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.forward(images)
        return outputs
