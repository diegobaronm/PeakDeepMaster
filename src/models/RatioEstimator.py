import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RatioEstimatorNet(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int = 1, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + theta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, theta], dim=1))


class LLHRatioEstimator(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])

        self.learning_rate = cfg.train.learning_rate
        self.weight_decay = cfg.train.weight_decay
        self.lr_patience = cfg.train.lr_patience
        self.lr_factor = cfg.train.lr_factor

        from src.data.DataHelpers import is_split_only, normalize_feature_specs

        parameter_specs = normalize_feature_specs(cfg.dataset.parameters)
        model_param_count = sum(1 for spec in parameter_specs if not is_split_only(spec))

        self.model = RatioEstimatorNet(
            x_dim=len(cfg.dataset.observables),
            theta_dim=model_param_count,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.x_column_indices = None
        self.parameter_column_indices = None
        self.parameter_column_index = None
        self.weight_column_index = None

    def setup(self, stage: str | None = None):
        dm = self.trainer.datamodule
        self.x_column_indices = list(dm.x_column_indices)
        self.parameter_column_indices = list(dm.parameter_column_indices)
        self.parameter_column_index = int(dm.parameter_column_index)
        self.weight_column_index = int(dm.weight_column_index)

    def _step(self, batch, stage: str):
        inputs, targets = batch
        y_true = targets[:, 0]

        x = inputs[:, self.x_column_indices]
        theta = inputs[:, self.parameter_column_indices]
        event_weights = torch.abs(inputs[:, self.weight_column_index])

        logits = self.model(x, theta).squeeze()
        loss_per_event = self.loss_fn(logits, y_true)
        loss = torch.sum(loss_per_event * event_weights)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss, logits, y_true, targets[:, 1]

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _ = self._step(batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y_true, _ = self._step(batch, stage="test")
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        acc = (pred == y_true).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        x = inputs[:, self.x_column_indices]
        theta = inputs[:, self.parameter_column_indices]
        logits = self.model(x, theta)
        return {
            "predictions": torch.sigmoid(logits),
            "labels": targets[:, 0],
            "categories": targets[:, 1],
            "weights": inputs[:, self.weight_column_index],
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
