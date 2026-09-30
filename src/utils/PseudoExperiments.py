"""Pseudo-experiment-based parameter uncertainty estimation."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from src.data.DataHelpers import parameter_point_label

logger = logging.getLogger(__name__)

_CONTOUR_PROBABILITIES = (("1-sigma", 0.6827), ("2-sigma", 0.9545))


def gaussian_contour_levels() -> list[tuple[str, float, float]]:
    """Return confidence labels, enclosed probabilities, and 2-D radii squared."""
    return [
        (label, probability, float(-2.0 * np.log1p(-probability)))
        for label, probability in _CONTOUR_PROBABILITIES
    ]


class PseudoExperimentEstimator:
    """Generate pseudo-experiments from a hypothesis histogram and estimate
    parameter uncertainties by scanning over pre-computed inference shapes."""

    def __init__(
        self,
        hypothesis_shape: np.ndarray,
        hypothesis_sigma: np.ndarray,
        n_pseudo: int = 1000,
        random_seed: int = 42,
        strategy: str = "gaussian",
    ):
        self.hypothesis_shape = hypothesis_shape
        self.hypothesis_sigma = hypothesis_sigma
        self.n_pseudo = n_pseudo
        self.strategy = strategy.lower()
        if self.strategy not in {"gaussian", "poisson"}:
            raise ValueError(f"Unknown pseudo-experiment strategy: {strategy}")
        self.rng = np.random.default_rng(random_seed)

        self.pseudo_experiments: np.ndarray | None = None
        self.scan_points: list[tuple[float, ...]] = []
        self.scan_shapes: list[np.ndarray] = []
        self.scan_sigmas: list[np.ndarray] = []
        self.best_fit_parameters: list[tuple[float, ...]] = []
        self.best_fit_chi2s: list[float] = []

    def generate(self) -> np.ndarray:
        """Generate pseudo-experiments by fluctuating the hypothesis histogram."""
        n_bins = len(self.hypothesis_shape)
        if self.strategy == "poisson":
            if not np.all(np.isfinite(self.hypothesis_shape)):
                raise ValueError("Poisson pseudo-experiments require finite bin contents.")
            counts = self.rng.poisson(
                np.abs(self.hypothesis_shape), size=(self.n_pseudo, n_bins)
            )
            self.pseudo_experiments = counts * np.sign(self.hypothesis_shape)
            logger.info(
                "Generated %d Poisson pseudo-experiments with %d bins each.",
                self.n_pseudo, n_bins,
            )
            return self.pseudo_experiments

        self.pseudo_experiments = np.empty((self.n_pseudo, n_bins))
        for i in range(self.n_pseudo):
            self.pseudo_experiments[i] = self.rng.normal(
                self.hypothesis_shape, self.hypothesis_sigma
            )
        logger.info(
            "Generated %d pseudo-experiments with %d bins each.",
            self.n_pseudo, n_bins,
        )
        return self.pseudo_experiments

    def add_scan_point(
        self,
        theta_point: tuple[float, ...],
        infer_shape: np.ndarray,
        infer_sigma: np.ndarray,
    ) -> None:
        """Store an inference shape for a parameter scan point."""
        self.scan_points.append(theta_point)
        self.scan_shapes.append(infer_shape.copy())
        self.scan_sigmas.append(infer_sigma.copy())

    def find_best_fits(self) -> np.ndarray:
        """For each pseudo-experiment, find the scan point that minimises L2."""
        if self.pseudo_experiments is None:
            raise RuntimeError("Call generate() before find_best_fits().")
        if len(self.scan_points) == 0:
            raise RuntimeError("No scan points stored. Call add_scan_point() during the scan.")

        scan_matrix = np.asarray(self.scan_shapes)  # (n_theta, n_bins)
        self.best_fit_parameters = []
        self.best_fit_chi2s = []

        for pseudo in self.pseudo_experiments:
            if self.strategy == "poisson":
                total = np.abs(pseudo).sum()
                if total > 0:
                    pseudo = pseudo / total
            diff = pseudo[np.newaxis, :] - scan_matrix
            chi2_per_point = np.sum(diff ** 2, axis=1)
            best_idx = int(np.argmin(chi2_per_point))
            self.best_fit_parameters.append(self.scan_points[best_idx])
            self.best_fit_chi2s.append(float(chi2_per_point[best_idx]))

        logger.info(
            "Computed best-fit parameters for %d pseudo-experiments.",
            len(self.best_fit_parameters),
        )
        return np.asarray(self.best_fit_parameters)

    def analyze_2d_best_fits(
        self,
        scan_axes: list[np.ndarray],
        parameter_names: list[str],
        parameter_labels: list[str],
        nominal_best_fit: tuple[float, float],
        truth_point: tuple[float, float],
        output_dir: Path,
        font_size: int = 14,
    ) -> dict[str, float]:
        """Build a 2-D best-fit grid and fit a zero-correlation Gaussian."""
        if len(scan_axes) != 2 or len(parameter_names) != 2 or len(parameter_labels) != 2:
            raise ValueError("2-D pseudo-experiment analysis requires exactly two parameters.")
        if len(nominal_best_fit) != 2 or len(truth_point) != 2:
            raise ValueError("Nominal best fit and truth point must each contain two values.")
        if self.n_pseudo <= 0 or len(self.best_fit_parameters) != self.n_pseudo:
            raise ValueError(
                "The number of best fits must equal the configured positive n_pseudo count."
            )

        axes = [np.asarray(axis, dtype=float) for axis in scan_axes]
        for dim, axis in enumerate(axes):
            if axis.ndim != 1 or len(np.unique(axis)) < 3 or not np.all(np.isfinite(axis)):
                raise ValueError(
                    f"Scan axis {dim} must contain at least three distinct finite points."
                )
            if np.any(np.diff(axis) <= 0):
                raise ValueError(f"Scan axis {dim} must be strictly increasing.")

        best_fits = np.asarray(self.best_fit_parameters, dtype=float)
        if best_fits.shape != (self.n_pseudo, 2) or not np.all(np.isfinite(best_fits)):
            raise ValueError("Best-fit parameters must be a finite (n_pseudo, 2) array.")
        for dim in range(2):
            if len(np.unique(best_fits[:, dim])) < 2:
                raise ValueError(
                    f"Best-fit samples have no spread on parameter axis {dim}; width is not identifiable."
                )

        # Keep the grid indexed as [parameter 1, parameter 2] for all outputs.
        counts = np.zeros((len(axes[0]), len(axes[1])), dtype=np.int64)
        for point in best_fits:
            indices = []
            for dim, (axis, value) in enumerate(zip(axes, point)):
                matches = np.flatnonzero(np.isclose(axis, value, rtol=1e-9, atol=1e-12))
                if len(matches) != 1:
                    raise ValueError(
                        f"Best-fit coordinate {value} is not on scan axis {dim}."
                    )
                indices.append(int(matches[0]))
            counts[tuple(indices)] += 1

        if int(counts.sum()) != self.n_pseudo:
            raise RuntimeError("The 2-D best-fit grid does not contain every pseudo-experiment.")

        # Normalize by the configured experiment count; fit only widths around the nominal best fit.
        normalized_counts = counts.astype(float) / self.n_pseudo
        x_grid, y_grid = np.meshgrid(axes[0], axes[1], indexing="ij")
        means = np.asarray(nominal_best_fit, dtype=float)
        if not np.all(np.isfinite(means)):
            raise ValueError("Nominal best-fit coordinates must be finite.")

        spacings = [float(np.min(np.diff(axis))) for axis in axes]
        spans = [float(axis[-1] - axis[0]) for axis in axes]
        lower_bounds = np.asarray(spacings) * 1e-3
        upper_bounds = np.asarray(spans) * 10.0
        if np.any(upper_bounds <= lower_bounds):
            raise ValueError("Scan axes do not provide a usable range for Gaussian widths.")

        # Treat the finite scan grid as a discrete probability surface.
        def probability_surface(log_sigmas: np.ndarray) -> np.ndarray:
            sigmas = np.exp(log_sigmas)
            exponent = -0.5 * (
                ((x_grid - means[0]) / sigmas[0]) ** 2
                + ((y_grid - means[1]) / sigmas[1]) ** 2
            )
            surface = np.exp(exponent)
            total = surface.sum()
            if not np.isfinite(total) or total <= 0:
                return np.full_like(surface, np.nan)
            return surface / total

        initial_sigmas = np.maximum(np.std(best_fits, axis=0), spacings)
        initial_sigmas = np.clip(initial_sigmas, lower_bounds * 2, upper_bounds / 2)
        fit = least_squares(
            lambda log_sigmas: (probability_surface(log_sigmas) - normalized_counts).ravel(),
            x0=np.log(initial_sigmas),
            bounds=(np.log(lower_bounds), np.log(upper_bounds)),
        )
        sigmas = np.exp(fit.x)
        if not fit.success or not np.all(np.isfinite(sigmas)):
            raise RuntimeError(f"2-D Gaussian width fit failed: {fit.message}")
        if np.any(sigmas <= lower_bounds * 1.01):
            raise RuntimeError("2-D Gaussian fit collapsed to a scan-resolution boundary.")

        fitted_surface = probability_surface(fit.x)
        residuals = fitted_surface - normalized_counts
        fit_result = {
            "mu_x": float(means[0]),
            "mu_y": float(means[1]),
            "sigma_x": float(sigmas[0]),
            "sigma_y": float(sigmas[1]),
            "n_pseudo": int(self.n_pseudo),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "sum_squared_residuals": float(np.sum(residuals ** 2)),
        }

        # Save the full grid and fit parameters so the plotted Z values are reproducible.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_rows = []
        for x_index, x_value in enumerate(axes[0]):
            for y_index, y_value in enumerate(axes[1]):
                grid_rows.append({
                    parameter_names[0]: x_value,
                    parameter_names[1]: y_value,
                    "count": counts[x_index, y_index],
                    "normalized_count": normalized_counts[x_index, y_index],
                })
        pd.DataFrame(grid_rows).to_csv(
            output_dir / "pseudo_experiment_best_fit_grid.csv", index=False
        )
        fit_row = {
            "parameter_x": parameter_names[0],
            "parameter_y": parameter_names[1],
            **fit_result,
        }
        for label, probability, radius_squared in gaussian_contour_levels():
            fit_row[f"{label.replace('-', '_')}_probability"] = probability
            fit_row[f"{label.replace('-', '_')}_radius_squared"] = radius_squared
        pd.DataFrame([fit_row]).to_csv(
            output_dir / "pseudo_experiment_2d_gaussian_fit.csv", index=False
        )

        x_edges = self._histogram_edges(axes[0])
        y_edges = self._histogram_edges(axes[1])
        self._plot_2d_best_fit_grid(
            axes=axes,
            x_edges=x_edges,
            y_edges=y_edges,
            values=counts,
            fitted_surface=fitted_surface,
            parameter_labels=parameter_labels,
            nominal_best_fit=nominal_best_fit,
            truth_point=truth_point,
            output_dir=output_dir,
            font_size=font_size,
            normalized=False,
        )
        self._plot_2d_best_fit_grid(
            axes=axes,
            x_edges=x_edges,
            y_edges=y_edges,
            values=normalized_counts,
            fitted_surface=fitted_surface,
            parameter_labels=parameter_labels,
            nominal_best_fit=nominal_best_fit,
            truth_point=truth_point,
            output_dir=output_dir,
            font_size=font_size,
            normalized=True,
        )
        logger.info(
            "2-D pseudo-experiment Gaussian: mu=(%.6g, %.6g), sigma=(%.6g, %.6g)",
            means[0], means[1], sigmas[0], sigmas[1],
        )
        return fit_result

    @staticmethod
    def _histogram_edges(axis: np.ndarray) -> np.ndarray:
        midpoints = 0.5 * (axis[:-1] + axis[1:])
        return np.concatenate(([axis[0]], midpoints, [axis[-1]]))

    @staticmethod
    def _plot_2d_best_fit_grid(
        axes: list[np.ndarray],
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        values: np.ndarray,
        fitted_surface: np.ndarray,
        parameter_labels: list[str],
        nominal_best_fit: tuple[float, float],
        truth_point: tuple[float, float],
        output_dir: Path,
        font_size: int,
        normalized: bool,
    ) -> None:
        plt.rcParams.update({"font.size": font_size})
        fig, ax = plt.subplots(figsize=(8, 6))
        mesh = ax.pcolormesh(
            x_edges, y_edges, values.T, shading="flat", cmap="viridis"
        )
        colorbar_label = "Normalized pseudo-experiment count" if normalized else "Pseudo-experiment count"
        fig.colorbar(mesh, ax=ax, label=colorbar_label)

        # Probability contours belong on the normalized frequency plot, not raw counts.
        contour_specs = gaussian_contour_levels()
        surface = fitted_surface.T
        contour_values = [
            float(surface.max() * np.exp(-spec[2] / 2.0))
            for spec in contour_specs
        ]
        contour_levels = [contour_values[1], contour_values[0]]
        visible = [level for level in contour_levels if surface.min() < level < surface.max()]
        contour_legend_handles = []
        if normalized and visible:
            line_styles = ["dotted" if level == contour_values[1] else "solid" for level in visible]
            contours = ax.contour(
                axes[0], axes[1], surface,
                levels=visible,
                colors="white",
                linewidths=1.5,
                linestyles=line_styles,
            )
            labels = {
                contour_values[1]: r"2-$\sigma$",
                contour_values[0]: r"1-$\sigma$",
            }
            contour_legend_handles = [
                Line2D(
                    [0], [0], color="white", linestyle=line_style, linewidth=1.5,
                    label=labels[level],
                    path_effects=[
                        path_effects.Stroke(linewidth=3, foreground="black"),
                        path_effects.Normal(),
                    ],
                )
                for level, line_style in zip(visible, line_styles)
            ]

        ax.scatter(
            nominal_best_fit[0], nominal_best_fit[1], color="red", marker="x",
            label="Nominal best fit", zorder=3,
        )
        ax.scatter(
            truth_point[0], truth_point[1], color="green", marker="o",
            label="Truth", zorder=3,
        )
        ax.set_xlabel(parameter_labels[0], fontsize=font_size)
        ax.set_ylabel(parameter_labels[1], fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        handles, labels = ax.get_legend_handles_labels()
        contour_legend_handles.sort(
            key=lambda handle: 0 if handle.get_label().startswith("1-sigma") else 1
        )
        handles.extend(contour_legend_handles)
        labels.extend(handle.get_label() for handle in contour_legend_handles)
        ax.legend(
            handles, labels, fontsize=font_size,
            ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0,
        )
        fig.subplots_adjust(top=0.78)
        filename = (
            "pseudo_experiment_best_fit_normalized_2d.pdf"
            if normalized else "pseudo_experiment_best_fit_counts_2d.pdf"
        )
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def estimate_uncertainty(
        self, confidence: float = 0.95,
    ) -> dict[int, dict[str, float]]:
        """Compute the central confidence interval per parameter dimension."""
        params = np.asarray(self.best_fit_parameters)
        if params.ndim == 1:
            params = params.reshape(-1, 1)

        alpha = (1.0 - confidence) / 2.0
        results = {}
        for dim in range(params.shape[1]):
            col = params[:, dim]
            lower = float(np.percentile(col, alpha * 100))
            upper = float(np.percentile(col, (1.0 - alpha) * 100))
            results[dim] = {
                "mean": float(np.mean(col)),
                "median": float(np.median(col)),
                "lower": lower,
                "upper": upper,
                "confidence": confidence,
            }
        return results

    def save(
        self,
        output_dir: Path,
        parameter_names: list[str],
    ) -> None:
        """Persist pseudo-experiments, inference shapes, and best-fit results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "pseudo_experiments.npy", self.pseudo_experiments)

        shape_rows = []
        for theta, shape, sigma in zip(
            self.scan_points, self.scan_shapes, self.scan_sigmas
        ):
            row = {name: value for name, value in zip(parameter_names, theta)}
            for b, (s, e) in enumerate(zip(shape, sigma)):
                row[f"bin_{b}_content"] = s
                row[f"bin_{b}_error"] = e
            shape_rows.append(row)
        pd.DataFrame(shape_rows).to_csv(
            output_dir / "inference_shapes.csv", index=False
        )

        fit_rows = []
        for theta, chi2 in zip(self.best_fit_parameters, self.best_fit_chi2s):
            row = {name: value for name, value in zip(parameter_names, theta)}
            row["chi2"] = chi2
            fit_rows.append(row)
        pd.DataFrame(fit_rows).to_csv(
            output_dir / "best_fit_parameters.csv", index=False
        )

        logger.info("Saved pseudo-experiment results to %s", output_dir)

    def plot(
        self,
        parameter_axis_labels: list[str],
        truth_point: tuple[float, ...],
        nominal_best_fit: tuple[float, ...],
        output_dir: Path,
        confidence: float = 0.95,
        font_size: int = 14,
        parameter_display_names: list[str] | None = None,
        parameter_units: list[str | None] | None = None,
        output_file_names: list[str] | None = None,
    ) -> None:
        """Plot histogram of best-fit parameters with uncertainty bands."""
        output_dir = Path(output_dir)
        params = np.asarray(self.best_fit_parameters)
        if params.ndim == 1:
            params = params.reshape(-1, 1)

        if parameter_display_names is None:
            parameter_display_names = parameter_axis_labels
        if output_file_names is None:
            output_file_names = parameter_display_names

        uncertainties = self.estimate_uncertainty(confidence)

        for dim, axis_label in enumerate(parameter_axis_labels):
            col = params[:, dim]
            info = uncertainties[dim]
            display_name = parameter_display_names[dim]
            output_file_name = output_file_names[dim]
            units = None if parameter_units is None else [parameter_units[dim]]
            truth_label = parameter_point_label([display_name], (truth_point[dim],), parameter_units=units)
            nominal_label = parameter_point_label([display_name], (nominal_best_fit[dim],), parameter_units=units)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(col, bins=50, edgecolor="black", alpha=0.7, label="Pseudo-experiments")
            ax.axvline(
                truth_point[dim], color="green", linestyle="--", linewidth=1.5,
                label=f"Truth ({truth_label})",
            )
            ax.axvline(
                nominal_best_fit[dim], color="red", linestyle="-", linewidth=1.5,
                label=f"Nominal best fit ({nominal_label})",
            )
            ax.axvspan(
                info["lower"], info["upper"], alpha=0.2, color="blue",
                label=f"{confidence:.0%} CI",
            )

            ax.set_xlabel(axis_label, fontsize=font_size)
            ax.set_ylabel("Pseudo-experiments", fontsize=font_size)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=font_size)

            fig.tight_layout()
            fig.savefig(output_dir / f"pseudo_experiment_{output_file_name}.pdf", dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(
                "Parameter '%s': %s CI = [%.4g, %.4g], median = %.4g",
                axis_label, f"{confidence:.0%}", info["lower"], info["upper"], info["median"],
            )
