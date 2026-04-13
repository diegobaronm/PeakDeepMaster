"""Pseudo-experiment-based parameter uncertainty estimation."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PseudoExperimentEstimator:
    """Generate pseudo-experiments from a hypothesis histogram and estimate
    parameter uncertainties by scanning over pre-computed inference shapes."""

    def __init__(
        self,
        hypothesis_shape: np.ndarray,
        hypothesis_sigma: np.ndarray,
        n_pseudo: int = 1000,
        random_seed: int = 42,
    ):
        self.hypothesis_shape = hypothesis_shape
        self.hypothesis_sigma = hypothesis_sigma
        self.n_pseudo = n_pseudo
        self.rng = np.random.default_rng(random_seed)

        self.pseudo_experiments: np.ndarray | None = None
        self.scan_points: list[tuple[float, ...]] = []
        self.scan_shapes: list[np.ndarray] = []
        self.scan_sigmas: list[np.ndarray] = []
        self.best_fit_parameters: list[tuple[float, ...]] = []
        self.best_fit_chi2s: list[float] = []

    def generate(self) -> np.ndarray:
        """Generate pseudo-experiments by Gaussian-fluctuating the hypothesis."""
        n_bins = len(self.hypothesis_shape)
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
        parameter_names: list[str],
        truth_point: tuple[float, ...],
        nominal_best_fit: tuple[float, ...],
        output_dir: Path,
        confidence: float = 0.95,
        font_size: int = 14,
    ) -> None:
        """Plot histogram of best-fit parameters with uncertainty bands."""
        output_dir = Path(output_dir)
        params = np.asarray(self.best_fit_parameters)
        if params.ndim == 1:
            params = params.reshape(-1, 1)

        uncertainties = self.estimate_uncertainty(confidence)

        for dim, name in enumerate(parameter_names):
            col = params[:, dim]
            info = uncertainties[dim]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(col, bins=50, edgecolor="black", alpha=0.7, label="Pseudo-experiments")
            ax.axvline(
                truth_point[dim], color="green", linestyle="--", linewidth=1.5,
                label=f"Truth = {truth_point[dim]:.4g}",
            )
            ax.axvline(
                nominal_best_fit[dim], color="red", linestyle="-", linewidth=1.5,
                label=f"Nominal best fit = {nominal_best_fit[dim]:.4g}",
            )
            ax.axvspan(
                info["lower"], info["upper"], alpha=0.2, color="blue",
                label=f"{confidence:.0%} CI: [{info['lower']:.4g}, {info['upper']:.4g}]",
            )

            ax.set_xlabel(name, fontsize=font_size)
            ax.set_ylabel("Pseudo-experiments", fontsize=font_size)
            ax.set_title(
                f"Best-fit {name} distribution ({self.n_pseudo} pseudo-experiments)",
                fontsize=font_size + 1,
            )
            ax.legend(fontsize=font_size)

            fig.tight_layout()
            fig.savefig(output_dir / f"pseudo_experiment_{name}.pdf", dpi=150)
            plt.close(fig)

            logger.info(
                "Parameter '%s': %s CI = [%.4g, %.4g], median = %.4g",
                name, f"{confidence:.0%}", info["lower"], info["upper"], info["median"],
            )
