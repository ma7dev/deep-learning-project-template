import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import cool_project.optims as optims
from torch.optim import Adam
from torchmetrics import Accuracy

from cool_project.models.baseline import Baseline


class LitModel(pl.LightningModule):
    def __init__(
        self: pl.LightningModule,
        model_config: dict = None,
        optimizer_config: dict = None,
        lr_config: dict = None,
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
                "acc": Accuracy(),
                # 'acc': Accuracy(num_classes=self.num_classes, average='macro'),
                # 'f1': F1Score(num_classes=self.num_classes, average='macro'),
                # 'prec': Precision(num_classes=self.num_classes, average='macro'),
                # 'rec': Recall(num_classes=self.num_classes, average='macro'),
            },
            "val": {
                "acc": Accuracy(),
                # 'acc': Accuracy(num_classes=self.num_classes, average='macro'),
                # 'f1': F1Score(num_classes=self.num_classes, average='macro'),
                # 'prec': Precision(num_classes=self.num_classes, average='macro'),
                # 'rec': Recall(num_classes=self.num_classes, average='macro'),
            },
            "test": {
                "acc": Accuracy(),
                # 'acc': Accuracy(num_classes=self.num_classes, average='macro'),
                # 'f1': F1Score(num_classes=self.num_classes, average='macro'),
                # 'prec': Precision(num_classes=self.num_classes, average='macro'),
                # 'rec': Recall(num_classes=self.num_classes, average='macro'),
            },
        }

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), **self.optimizer_config.kwargs)
        return optimizer

    def forward(self, x):
        return self.model(x)

    # logging
    def logging_step(self, loss, acc, mode):
        self.log(f"{mode}/step/loss", loss, on_step=True, rank_zero_only=True)
        for metric_name, ac in acc.items():
            self.log(
                f"{mode}/step/{metric_name}",
                ac,
                on_step=True,
                rank_zero_only=True,
            )

    def logging_epoch(self, outputs, mode):
        loss = torch.stack([x["loss"].detach() for x in outputs]).mean()
        self.log(
            f"{mode}/epoch/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        for metric_name in self.metrics[mode].keys():
            try:
                acc = self.metrics[mode][metric_name].compute()
                self.log(
                    f"{mode}/epoch/{metric_name}",
                    acc,
                    on_epoch=True,
                    prog_bar=True,
                    rank_zero_only=True,
                )
                self.metrics[mode][metric_name].reset()
            except Exception as e:
                print(e)
                __import__("pdb").set_trace()

    # compute
    def compute_metrics(self, preds, targets, mode):
        preds = preds.detach().cpu().int()
        targets = targets.detach().cpu().int()

        acc = {}
        for metric_name in self.metrics[mode].keys():
            acc[metric_name] = self.metrics[mode][metric_name](preds, targets)
        return acc

    def _step(self, batch, batch_idx, mode=None):
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
            with torch.no_grad():
                outputs = self(images)
                loss = self.criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

        return {
            "loss": loss.clone(),
            "preds": preds.clone().detach(),
            "targets": targets.clone().detach(),
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def _end_step(self, step_outputs, mode=None):
        loss = step_outputs["loss"].detach()
        acc = self.compute_metrics(
            step_outputs["preds"], step_outputs["targets"], mode
        )
        self.logging_step(loss, acc, mode)

    def training_end_step(self, step_outputs):
        return self._end_step(step_outputs, "train")

    def validation_end_step(self, step_outputs):
        return self._end_step(step_outputs, "val")

    def test_end_step(self, step_outputs):
        return self._end_step(step_outputs, "test")

    # _epoch_end
    def _epoch_end(self, epoch_outputs, mode):
        # if mode == 'train':
        #     lr_scheduler = self.lr_schedulers()
        #     lr_scheduler.step()
        self.logging_epoch(epoch_outputs, mode)
        # pass

    def training_epoch_end(self, epoch_outputs):
        self._epoch_end(epoch_outputs, "train")

    def validation_epoch_end(self, epoch_outputs):
        self._epoch_end(epoch_outputs, "val")

    def test_epoch_end(self, epoch_outputs):
        self._epoch_end(epoch_outputs, "test")

    def predict_step(self, images):
        outputs = self.forward(images)
        return outputs
