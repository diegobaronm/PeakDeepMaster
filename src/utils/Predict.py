import lightning as L
import pandas as pd
from omegaconf import DictConfig

from src.data.DataHelpers import parameter_point_label
from src.utils.utils import (
    ensure_parent_dir,
    get_latest_checkpoint_path,
    load_checkpoint_into_model,
    resolve_runtime_path,
    set_execution_device,
)


def predict(datamodule, model_class, cfg: DictConfig) -> None:
    device = set_execution_device(cfg.general.device)
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)

    model = model_class(cfg)
    datamodule.setup(stage="predict")

    ckpt_path = get_latest_checkpoint_path(cfg.predict.checkpoint_path)
    model = load_checkpoint_into_model(model, ckpt_path)
    model.eval()

    outputs = trainer.predict(model, datamodule=datamodule)

    rows = []
    category_to_parameter_point = datamodule.category_to_parameter_point
    for out in outputs:
        preds = out["predictions"].cpu().numpy().reshape(-1)
        labels = out["labels"].cpu().numpy().reshape(-1)
        cats = out["categories"].cpu().numpy().reshape(-1)
        for p, y, c in zip(preds, labels, cats):
            category = int(c)
            parameter_point = category_to_parameter_point.get(category)
            rows.append(
                {
                    "prediction": float(p),
                    "label": float(y),
                    "category": category,
                    "parameter_point": None
                    if parameter_point is None
                    else parameter_point_label(datamodule.parameter_names, parameter_point),
                }
            )

    output_file = resolve_runtime_path(cfg.predict.output_file)
    ensure_parent_dir(output_file)
    pd.DataFrame(rows).to_csv(output_file, index=False)
