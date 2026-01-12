from typing import Tuple

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score, Precision, Recall, MetricCollection
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ECGClassifier(LightningModule):
    def __init__(self, lr: float = 1e-3, num_classes: int = 3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes

        # Load EfficientNet-B0
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify first conv layer to accept 1 channel instead of 3
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        old_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights for the new conv layer
        # Can Average the weights of the original 3 channels to initialize the single channel
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(old_conv.weight, dim=1, keepdim=True)

        self.model.features[0][0] = new_conv

        # Modify classifier to output num_classes
        # EfficientNet classifier is a Sequential, valid for B0: Dropout -> Linear
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "prec": Precision(task="multiclass", num_classes=num_classes, average="macro"),
                "rec": Recall(task="multiclass", num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_metrics(logits, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_metrics(logits, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss)
        self.test_metrics(logits, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    model = ECGClassifier()
    print(model)
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
