from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from src.data.DataHelpers import build_label_skip_mask, build_parameter_label_map, build_parameter_units_map, normalize_feature_specs, parameter_point_label, parameter_point_slug
from src.utils.utils import (
    ensure_parent_dir,
    get_latest_checkpoint_path,
    load_checkpoint_into_model,
    resolve_runtime_path,
    set_execution_device,
)


def _build_parameter_x_labels(cfg: DictConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    variables = getattr(cfg, "input_plots", {}).get("variables", [])
    for var in variables:
        mapping[var["name"]] = var.get("x_label", var["name"])
    return build_parameter_label_map(normalize_feature_specs(cfg.dataset.parameters), mapping, include_units=False)


def _build_parameter_units(cfg: DictConfig) -> dict[str, str | None]:
    return build_parameter_units_map(normalize_feature_specs(cfg.dataset.parameters))


def _category_to_parameter_map(parameters_category_dict: dict[tuple[float, ...], int]) -> dict[int, tuple[float, ...]]:
    return {int(category): tuple(parameter) for parameter, category in parameters_category_dict.items()}


def _category_label(
    category_to_parameter: dict[int, tuple[float, ...]],
    parameter_names: list[str],
    category: int,
    parameter_x_labels: dict[str, str] | None = None,
    parameter_units: dict[str, str | None] | None = None,
    label_skip_mask: list[bool] | None = None,
) -> str:
    parameter_point = category_to_parameter.get(category)
    if parameter_point is None:
        return f"category={category}"
    if parameter_x_labels:
        display_names = [parameter_x_labels.get(n, n) for n in parameter_names]
        units = None if parameter_units is None else [parameter_units.get(n) for n in parameter_names]
        return parameter_point_label(display_names, parameter_point, skip_mask=label_skip_mask, parameter_units=units)
    units = None if parameter_units is None else [parameter_units.get(n) for n in parameter_names]
    return parameter_point_label(parameter_names, parameter_point, skip_mask=label_skip_mask, parameter_units=units)


def _category_slug(category_to_parameter: dict[int, tuple[float, ...]], parameter_names: list[str], category: int, label_skip_mask: list[bool] | None = None) -> str:
    parameter_point = category_to_parameter.get(category)
    if parameter_point is None:
        return f"category_{category}"
    return parameter_point_slug(parameter_names, parameter_point, skip_mask=label_skip_mask)


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
    category_to_parameter: dict[int, tuple[float, ...]],
    parameter_names: list[str],
    split_name: str,
    output_dir: Path,
    parameter_x_labels: dict[str, str] | None = None,
    parameter_units: dict[str, str | None] | None = None,
    font_size: int = 14,
    label_skip_mask: list[bool] | None = None,
) -> None:
    signal_categories = sorted({int(cat) for cat in categories[labels == 1]})
    if len(signal_categories) == 0:
        return

    ensure_parent_dir(str(output_dir / "placeholder.txt"))

    plt.rcParams.update({"font.size": font_size})
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

        label = _category_label(category_to_parameter, parameter_names, category, parameter_x_labels, parameter_units, label_skip_mask)
        plt.hist(signal_scores, bins=20, histtype="step", density=True, range=(0.0, 1.0))
        legend_labels.append(f"S {label}")

    plt.xlabel("Predicted Scores", fontsize=font_size)
    plt.ylabel("Density", fontsize=font_size)
    plt.yscale("log")
    plt.tick_params(labelsize=font_size)
    plt.legend(legend_labels, bbox_to_anchor=(1.35, 1.0), loc="upper right", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(output_dir / f"scores_{split_name}_by_category.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    by_category_dir = output_dir / "by_category"
    by_category_dir.mkdir(parents=True, exist_ok=True)
    background_scores = scores[labels == 0]
    for category in signal_categories:
        signal_scores = scores[categories == category]
        if signal_scores.size == 0:
            continue

        label = _category_label(category_to_parameter, parameter_names, category, parameter_x_labels, parameter_units, label_skip_mask)
        slug = _category_slug(category_to_parameter, parameter_names, category, label_skip_mask)
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
        plt.hist(signal_scores, bins=20, histtype="step", density=True, range=(0.0, 1.0), label=f"Signal {label}")
        plt.xlabel("Predicted Scores", fontsize=font_size)
        plt.ylabel("Density", fontsize=font_size)
        plt.yscale("log")
        plt.tick_params(labelsize=font_size)
        plt.legend(loc="upper right", fontsize=font_size)
        plt.tight_layout()
        plt.savefig(by_category_dir / f"scores_{split_name}_{slug}.pdf", dpi=200, bbox_inches="tight")
        plt.close()


def _plot_roc_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    category_to_parameter: dict[int, tuple[float, ...]],
    parameter_names: list[str],
    split_name: str,
    output_dir: Path,
    parameter_x_labels: dict[str, str] | None = None,
    parameter_units: dict[str, str | None] | None = None,
    font_size: int = 14,
    label_skip_mask: list[bool] | None = None,
) -> None:
    signal_categories = sorted({int(cat) for cat in categories[labels == 1]})
    if len(signal_categories) == 0:
        return

    roc_rows: list[dict[str, float | int | str]] = []
    plt.rcParams.update({"font.size": font_size})
    plt.figure(figsize=(8, 6))
    plt.plot([0.0, 1.0], [0.0, 1.0], color="navy", lw=2, linestyle="--")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate", fontsize=font_size)
    plt.ylabel("True Positive Rate", fontsize=font_size)
    plt.tick_params(labelsize=font_size)

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
        label = _category_label(category_to_parameter, parameter_names, category, parameter_x_labels, parameter_units, label_skip_mask)
        slug = _category_slug(category_to_parameter, parameter_names, category, label_skip_mask)

        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f} -- {label}")
        roc_rows.append(
            {
                "split": split_name,
                "category": int(category),
                "parameter_point": label,
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
        plt.xlabel("False Positive Rate", fontsize=font_size)
        plt.ylabel("True Positive Rate", fontsize=font_size)
        plt.tick_params(labelsize=font_size)
        plt.legend(loc="lower right", fontsize=font_size)
        plt.tight_layout()
        plt.savefig(by_category_dir / f"roc_{split_name}_{slug}.pdf", dpi=200, bbox_inches="tight")
        plt.close(plt_single)

    plt.legend(loc="lower right", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{split_name}_by_category.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    if roc_rows:
        pd.DataFrame(roc_rows).to_csv(output_dir / f"roc_summary_{split_name}.csv", index=False)


def _run_plots_for_split(
    trainer,
    model,
    datamodule,
    split_name: str,
    category_to_parameter: dict[int, tuple[float, ...]],
    output_dir: Path,
    batch_size: int,
    parameter_x_labels: dict[str, str] | None = None,
    parameter_units: dict[str, str | None] | None = None,
    font_size: int = 14,
    label_skip_mask: list[bool] | None = None,
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
        parameter_names=datamodule.parameter_names,
        split_name=split_name,
        output_dir=split_output_dir,
        parameter_x_labels=parameter_x_labels,
        parameter_units=parameter_units,
        font_size=font_size,
        label_skip_mask=label_skip_mask,
    )
    _plot_roc_curves(
        scores=predictions["scores"],
        labels=predictions["labels"],
        categories=predictions["categories"],
        category_to_parameter=category_to_parameter,
        parameter_names=datamodule.parameter_names,
        split_name=split_name,
        output_dir=split_output_dir,
        parameter_x_labels=parameter_x_labels,
        parameter_units=parameter_units,
        font_size=font_size,
        label_skip_mask=label_skip_mask,
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
    parameter_x_labels = _build_parameter_x_labels(cfg)
    parameter_units = _build_parameter_units(cfg)
    font_size = int(getattr(cfg.input_plots, "font_size", 14))
    label_skip_mask = build_label_skip_mask(normalize_feature_specs(cfg.dataset.parameters))
    _run_plots_for_split(trainer, model, datamodule, "test", category_to_parameter, output_dir, batch_size, parameter_x_labels, parameter_units, font_size, label_skip_mask)
    _run_plots_for_split(trainer, model, datamodule, "holdout", category_to_parameter, output_dir, batch_size, parameter_x_labels, parameter_units, font_size, label_skip_mask)
