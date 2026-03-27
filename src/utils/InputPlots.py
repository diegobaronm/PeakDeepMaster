"""Plot input variable distributions per parameter point."""
from pathlib import Path
import logging

import h5py
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from src.data.DataHelpers import (
    build_indices_per_parameter_point,
    get_unique_parameter_points,
    normalize_feature_specs,
    parameter_name_from_spec,
    parse_feature_spec,
)
from src.utils.utils import resolve_runtime_path

logger = logging.getLogger(__name__)


def _compare_distributions(
    variable: np.ndarray,
    weights: np.ndarray,
    indices_per_point: dict[tuple[float, ...], np.ndarray],
    parameter_names: list[str],
    x_range: list[float],
    variable_name: str,
    n_bins: int,
    use_weights: bool,
    density: bool,
    x_label: str | None,
    y_label: str | None,
    output_path: Path,
):
    """Draw overlaid histograms of *variable* for each parameter point."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    for point, indices in indices_per_point.items():
        label = ", ".join(
            f"{name}={value:g}" for name, value in zip(parameter_names, point)
        )
        data = variable[indices]
        w = weights[indices] if use_weights else None

        counts, bin_edges = np.histogram(
            data, bins=n_bins, range=x_range, weights=w,
        )

        if density:
            bin_widths = np.diff(bin_edges)
            abs_area = np.sum(np.abs(counts) * bin_widths)
            if abs_area > 0:
                counts = counts / abs_area

        ax.stairs(counts, bin_edges, label=label)

    ax.set_xlabel(x_label if x_label is not None else variable_name)
    ax.set_xlim(x_range)
    ax.set_ylabel(y_label if y_label is not None else ("Density" if density else "Events"))
    ax.set_title(f"Distribution of {variable_name}")
    ax.legend(fontsize="small", ncol=max(1, len(indices_per_point) // 6))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def _signal_plus_background_distributions(
    variable: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    indices_per_point: dict[tuple[float, ...], np.ndarray],
    parameter_names: list[str],
    x_range: list[float],
    variable_name: str,
    n_bins: int,
    use_weights: bool,
    x_label: str | None,
    y_label: str | None,
    output_dir: Path,
    file_prefix: str,
    log_scale: bool = False,
):
    """For each signal parameter point, plot signal+BG combined as density."""
    # Collect all background indices across parameter points.
    bg_indices_list = []
    signal_point_indices: dict[tuple[float, ...], np.ndarray] = {}
    for point, indices in indices_per_point.items():
        point_labels = labels[indices]
        point_bg = indices[point_labels == 0]
        point_sig = indices[point_labels == 1]
        if len(point_bg) > 0:
            bg_indices_list.append(point_bg)
        if len(point_sig) > 0:
            signal_point_indices[point] = point_sig

    if len(bg_indices_list) == 0 or len(signal_point_indices) == 0:
        logger.warning("Not enough signal/BG events to produce S+BG plots for %s", variable_name)
        return

    bg_indices = np.concatenate(bg_indices_list)

    # Build the BG histogram once.
    bg_w = weights[bg_indices] if use_weights else None
    bg_counts, bin_edges = np.histogram(
        variable[bg_indices], bins=n_bins, range=x_range, weights=bg_w,
    )

    for point, sig_idx in signal_point_indices.items():
        point_label = ", ".join(
            f"{name}={value:g}" for name, value in zip(parameter_names, point)
        )
        sig_w = weights[sig_idx] if use_weights else None
        sig_counts, _ = np.histogram(
            variable[sig_idx], bins=n_bins, range=x_range, weights=sig_w,
        )

        combined = bg_counts + sig_counts
        bin_widths = np.diff(bin_edges)

        # Normalise each to density using absolute area.
        bg_density = bg_counts.copy()
        abs_area_bg = np.sum(np.abs(bg_density) * bin_widths)
        if abs_area_bg > 0:
            bg_density = bg_density / abs_area_bg

        combined_density = combined.copy()
        abs_area_comb = np.sum(np.abs(combined_density) * bin_widths)
        if abs_area_comb > 0:
            combined_density = combined_density / abs_area_comb

        slug = "_".join(f"{v:g}" for v in point).replace(".", "p").replace("-", "m")

        # Always produce the linear-scale plot.
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.stairs(bg_density, bin_edges, label="Background")
        ax.stairs(combined_density, bin_edges, label=f"S+BG ({point_label})")
        ax.set_xlabel(x_label if x_label is not None else variable_name)
        ax.set_xlim(x_range)
        ax.set_ylabel(y_label if y_label is not None else "Density")
        ax.set_title(f"S+BG distribution of {variable_name} ({point_label})")
        ax.legend(fontsize="small")
        fig.tight_layout()
        out_path = output_dir / f"{file_prefix}_splusbg_{slug}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        logger.info("Saved %s", out_path)

        # Optionally produce a log-scale version as well.
        if log_scale:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.stairs(bg_density, bin_edges, label="Background")
            ax.stairs(combined_density, bin_edges, label=f"S+BG ({point_label})")
            ax.set_xlabel(x_label if x_label is not None else variable_name)
            ax.set_xlim(x_range)
            ax.set_ylabel(y_label if y_label is not None else "Density")
            ax.set_title(f"S+BG distribution of {variable_name} ({point_label})")
            ax.set_yscale("log")
            ax.legend(fontsize="small")
            fig.tight_layout()
            out_path = output_dir / f"{file_prefix}_splusbg_{slug}_log.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            logger.info("Saved %s", out_path)


def run_input_plots(cfg: DictConfig) -> None:
    """Generate per-parameter-point distribution plots for all configured variables."""
    input_h5_path = resolve_runtime_path(cfg.dataset.input_h5_path)
    output_dir = Path(
        resolve_runtime_path(getattr(cfg.input_plots, "output_dir", "results/input_plots"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bins = int(getattr(cfg.input_plots, "n_bins", 100))
    density = bool(getattr(cfg.input_plots, "density", False))
    signal_plus_bg = bool(getattr(cfg.input_plots, "signal_plus_background", False))
    log_scale = bool(getattr(cfg.input_plots, "log_scale", False))
    global_x_label = getattr(cfg.input_plots, "x_label", None)
    global_y_label = getattr(cfg.input_plots, "y_label", None)
    max_events = int(cfg.dataset.max_events_per_parameter)
    seed = int(cfg.general.seed)

    parameter_specs = normalize_feature_specs(cfg.dataset.parameters)
    parameter_names = [parameter_name_from_spec(spec) for spec in parameter_specs]

    # Build the variable list to plot from the config.
    plot_vars = list(cfg.input_plots.variables)

    with h5py.File(input_h5_path, "r") as data_file:
        # Build parameter matrix and indices per point.
        parameter_arrays = []
        for spec in parameter_specs:
            group, variable = parse_feature_spec(spec)
            parameter_arrays.append(data_file["INPUTS"][group][variable][:])
        parameter_matrix = np.column_stack(parameter_arrays)
        parameter_points = get_unique_parameter_points(parameter_matrix)

        indices_per_point = build_indices_per_parameter_point(
            parameter_matrix=parameter_matrix,
            parameter_points=parameter_points,
            max_events_per_parameter=max_events,
            random_seed=seed,
        )

        # Load the weight array once.
        weight_specs = normalize_feature_specs(cfg.dataset.weights)
        w_group, w_variable = parse_feature_spec(weight_specs[0])
        weights = data_file["INPUTS"][w_group][w_variable][:]

        # Load class labels for S+BG plots.
        labels = data_file["LABELS"]["CLASS"][:].flatten() if signal_plus_bg else None

        for var_cfg in plot_vars:
            group = str(var_cfg.group)
            name = str(var_cfg.name)
            x_min = float(var_cfg.x_min)
            x_max = float(var_cfg.x_max)
            display_name = str(getattr(var_cfg, "display_name", name))
            use_weights = bool(getattr(var_cfg, "use_weights", True))
            var_n_bins = int(getattr(var_cfg, "n_bins", n_bins))
            var_density = bool(getattr(var_cfg, "density", density))
            var_x_label = getattr(var_cfg, "x_label", global_x_label)
            var_y_label = getattr(var_cfg, "y_label", global_y_label)
            var_log_scale = bool(getattr(var_cfg, "log_scale", log_scale))
            if var_x_label is not None:
                var_x_label = str(var_x_label)
            if var_y_label is not None:
                var_y_label = str(var_y_label)

            logger.info("Plotting %s/%s ...", group, name)
            values = data_file["INPUTS"][group][name][:]

            safe_name = f"{group}_{name}".replace("/", "_")
            output_path = output_dir / f"{safe_name}.pdf"

            _compare_distributions(
                variable=values,
                weights=weights,
                indices_per_point=indices_per_point,
                parameter_names=parameter_names,
                x_range=[x_min, x_max],
                variable_name=display_name,
                n_bins=var_n_bins,
                use_weights=use_weights,
                density=var_density,
                x_label=var_x_label,
                y_label=var_y_label,
                output_path=output_path,
            )

            if signal_plus_bg and use_weights:
                _signal_plus_background_distributions(
                    variable=values,
                    weights=weights,
                    labels=labels,
                    indices_per_point=indices_per_point,
                    parameter_names=parameter_names,
                    x_range=[x_min, x_max],
                    variable_name=display_name,
                    n_bins=var_n_bins,
                    use_weights=True,
                    x_label=var_x_label,
                    y_label=var_y_label,
                    output_dir=output_dir,
                    file_prefix=safe_name,
                    log_scale=var_log_scale,
                )

    logger.info("All input plots saved to %s", output_dir)
