import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src.utils.utils import set_execution_device, set_seed


def train(datamodule, model_class, cfg: DictConfig) -> None:
    set_seed(cfg.general.seed)
    device = set_execution_device(cfg.general.device)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, mode="min"),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ]

    model = model_class(cfg)
    if bool(cfg.train.get("compile", False)):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    train_logger = TensorBoardLogger(
        save_dir=cfg.logging.train.output_dir,
        name=cfg.logging.train.experiment_name,
    )

    trainer = L.Trainer(
        max_epochs=cfg.train.n_epochs,
        accelerator=device,
        devices="auto",
        callbacks=callbacks,
        logger=train_logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
