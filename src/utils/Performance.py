from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.utils.utils import (
    ensure_parent_dir,
    get_latest_checkpoint_path,
    load_checkpoint_into_model,
    resolve_runtime_path,
    set_execution_device,
)


def _category_to_parameter_map(parameters_category_dict: dict[float, int]) -> dict[int, float]:
    return {int(category): float(parameter) for parameter, category in parameters_category_dict.items()}


def _collect_predictions(trainer, model, dataset, batch_size: int) -> dict[str, np.ndarray] | None:
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = trainer.predict(model, dataloaders=loader)
    if not outputs:
        return None

    scores = np.concatenate([out["predictions"].detach().cpu().numpy().reshape(-1) for out in outputs])
    labels = np.concatenate([out["labels"].detach().cpu().numpy().reshape(-1) for out in outputs])
    categories = np.concatenate([out["categories"].detach().cpu().numpy().reshape(-1) for out in outputs]).astype(int)
    weights = np.concatenate([out["weights"].detach().cpu().numpy().reshape(-1) for out in outputs])

    return {
        "scores": scores,
        "labels": labels,
        "categories": categories,
        "weights": weights,
    }


def _plot_score_distributions(
    scores: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    category_to_parameter: dict[int, float],
    split_name: str,
    output_dir: Path,
) -> None:
    signal_categories = sorted({int(cat) for cat in categories[labels == 1]})
    if len(signal_categories) == 0:
        return

    ensure_parent_dir(str(output_dir / "placeholder.txt"))

    plt.figure(figsize=(9, 6))
    plt.hist(
        scores[labels == 0],
        bins=20,
        histtype="step",
        label="Background",
        density=True,
        color="black",
        linewidth=2.5,
        range=(0.0, 1.0),
    )

    legend_labels = ["BG"]
    for category in signal_categories:
        signal_scores = scores[categories == category]
        if signal_scores.size == 0:
            continue

        parameter = category_to_parameter.get(category, float(category))
        plt.hist(signal_scores, bins=20, histtype="step", density=True, range=(0.0, 1.0))
        legend_labels.append(f"S c={parameter:.2f}")

    plt.xlabel("Predicted Scores")
    plt.ylabel("Density")
    plt.title(f"{split_name.title()} score distributions by category")
    plt.yscale("log")
    plt.legend(legend_labels, bbox_to_anchor=(1.35, 1.0), loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"scores_{split_name}_by_category.png", dpi=200, bbox_inches="tight")
    plt.close()

    by_category_dir = output_dir / "by_category"
    by_category_dir.mkdir(parents=True, exist_ok=True)
    background_scores = scores[labels == 0]
    for category in signal_categories:
        signal_scores = scores[categories == category]
        if signal_scores.size == 0:
            continue

        parameter = category_to_parameter.get(category, float(category))
        plt.figure(figsize=(8, 5))
        plt.hist(
            background_scores,
            bins=20,
            histtype="step",
            label="Background",
            density=True,
            color="black",
            linewidth=2.5,
            range=(0.0, 1.0),
        )
        plt.hist(signal_scores, bins=20, histtype="step", density=True, range=(0.0, 1.0), label=f"Signal c={parameter:.2f}")
        plt.xlabel("Predicted Scores")
        plt.ylabel("Density")
        plt.title(f"{split_name.title()} scores: background vs c={parameter:.2f}")
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(by_category_dir / f"scores_{split_name}_c{parameter:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()


def _plot_roc_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    category_to_parameter: dict[int, float],
    split_name: str,
    output_dir: Path,
) -> None:
    signal_categories = sorted({int(cat) for cat in categories[labels == 1]})
    if len(signal_categories) == 0:
        return

    roc_rows: list[dict[str, float | int | str]] = []
    plt.figure(figsize=(8, 6))
    plt.plot([0.0, 1.0], [0.0, 1.0], color="navy", lw=2, linestyle="--")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name.title()} ROC by signal category")

    by_category_dir = output_dir / "by_category"
    by_category_dir.mkdir(parents=True, exist_ok=True)
    background_mask = labels == 0

    for category in signal_categories:
        category_mask = categories == category
        selection_mask = background_mask | category_mask
        selected_labels = labels[selection_mask]
        selected_scores = scores[selection_mask]
        if np.unique(selected_labels).size < 2:
            continue

        fpr, tpr, _ = roc_curve(selected_labels, selected_scores)
        roc_auc = auc(fpr, tpr)
        parameter = category_to_parameter.get(category, float(category))

        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f} -- c = {parameter:.2f}")
        roc_rows.append(
            {
                "split": split_name,
                "category": int(category),
                "parameter": parameter,
                "auc": float(roc_auc),
                "signal_count": int(category_mask.sum()),
                "background_count": int(background_mask.sum()),
            }
        )

        plt_single = plt.figure(figsize=(7, 5))
        plt.plot([0.0, 1.0], [0.0, 1.0], color="navy", lw=2, linestyle="--")
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split_name.title()} ROC: background vs c={parameter:.2f}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(by_category_dir / f"roc_{split_name}_c{parameter:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close(plt_single)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{split_name}_by_category.png", dpi=200, bbox_inches="tight")
    plt.close()

    if roc_rows:
        pd.DataFrame(roc_rows).to_csv(output_dir / f"roc_summary_{split_name}.csv", index=False)


def _run_plots_for_split(
    trainer,
    model,
    datamodule,
    split_name: str,
    category_to_parameter: dict[int, float],
    output_dir: Path,
    batch_size: int,
) -> None:
    predictions = _collect_predictions(
        trainer=trainer,
        model=model,
        dataset=datamodule.get_eval_dataset(split_name),
        batch_size=batch_size,
    )
    if predictions is None:
        return

    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    _plot_score_distributions(
        scores=predictions["scores"],
        labels=predictions["labels"],
        categories=predictions["categories"],
        category_to_parameter=category_to_parameter,
        split_name=split_name,
        output_dir=split_output_dir,
    )
    _plot_roc_curves(
        scores=predictions["scores"],
        labels=predictions["labels"],
        categories=predictions["categories"],
        category_to_parameter=category_to_parameter,
        split_name=split_name,
        output_dir=split_output_dir,
    )


def testing(datamodule, model_class, cfg: DictConfig) -> None:
    device = set_execution_device(cfg.general.device)
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)

    model = model_class(cfg)
    datamodule.setup(stage="test")

    ckpt_path = get_latest_checkpoint_path(cfg.performance.checkpoint_path)
    model = load_checkpoint_into_model(model, ckpt_path)
    model.eval()

    trainer.test(model, datamodule=datamodule)

    batch_size = int(getattr(cfg.performance, "batch_size", datamodule.val_batch_size))
    output_dir = Path(resolve_runtime_path(getattr(cfg.performance, "output_dir", "results/performance")))
    output_dir.mkdir(parents=True, exist_ok=True)

    category_to_parameter = _category_to_parameter_map(datamodule.parameters_category_dict)
    _run_plots_for_split(trainer, model, datamodule, "test", category_to_parameter, output_dir, batch_size)
    _run_plots_for_split(trainer, model, datamodule, "holdout", category_to_parameter, output_dir, batch_size)
