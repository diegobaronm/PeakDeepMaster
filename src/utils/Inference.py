from pathlib import Path
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src.data.DataHelpers import feature_key, parameter_point_label, parse_feature_spec
from src.utils.utils import get_latest_checkpoint_path, load_checkpoint_into_model, resolve_runtime_path

logger = logging.getLogger(__name__)

def logits(x_obs: torch.Tensor, theta: torch.Tensor, model) -> torch.Tensor:
    return model(x_obs, theta).reshape(-1)


def likelihood_ratio(theta: torch.Tensor, x_obs: torch.Tensor, model) -> torch.Tensor:
    thetas = theta.unsqueeze(0).expand(len(x_obs), -1)
    scores = torch.sigmoid(logits(x_obs, thetas, model))
    return scores / (1 - scores + 1e-8)


def rosmm(theta: torch.Tensor, x_obs: torch.Tensor, model_pp, model_pn, c_zero: float, c_one: float) -> torch.Tensor:
    llr_pp = likelihood_ratio(theta, x_obs, model_pp)
    llr_pn = likelihood_ratio(theta, x_obs, model_pn)
    return llr_pp * (c_one / c_zero) + llr_pn * ((1 - c_one) / c_zero)


def chi_squared(observed: np.ndarray, expected: np.ndarray, expected_variance: np.ndarray) -> float:
    return float(np.sum(((observed - expected) ** 2) / expected_variance))


def _hist(values: np.ndarray, weights: np.ndarray, n_bins: int = 50, x_min: float = 500.0, x_max: float = 1200.0):
    hist, edges = np.histogram(values, bins=n_bins, range=(x_min, x_max), weights=weights)
    var_hist, _ = np.histogram(values, bins=n_bins, range=(x_min, x_max), weights=weights ** 2)
    return hist, var_hist, edges


def hypothesis_plot(hypothesis_shape: np.ndarray, hypothesis_var: np.ndarray, hypothesis_edges: np.ndarray, observable: str, output_dir: Path):
    logger.debug(hypothesis_shape)
    logger.debug(hypothesis_var)
    logger.debug(hypothesis_edges)
    plt.clf()
    plt.stairs(hypothesis_shape, hypothesis_edges, label="Hypothesis")
    plt.title("Hypothesis normalised shape")
    plt.xlabel("%s [GeV]" % observable)
    plt.ylabel("Normalised events")
    plt.legend()
    plt.savefig(output_dir / "hypothesis_shape.pdf")


def inference_scan_plot(infer_shape: np.ndarray, hypothesis_shape: np.ndarray, hypothesis_edges: np.ndarray, observable: str, theta_label: str, output_dir: Path, postfix: int):
    plt.clf()
    plt.stairs(infer_shape, hypothesis_edges, label="Inference")
    plt.stairs(hypothesis_shape, hypothesis_edges, label="Hypothesis")
    plt.title(f"Inference shape {theta_label}")
    plt.xlabel("%s [GeV]" % observable)
    plt.ylabel("Normalised events")
    plt.legend()
    plot_name = "inference_scan_%d.pdf" % postfix
    plt.savefig(output_dir / plot_name)


def chi2_plot(theta_scan: np.ndarray, chi2_values: np.ndarray, param_variable: str, output_dir: Path):
    plt.clf()
    plt.plot(theta_scan, chi2_values)
    plt.title(r"$\chi^2$ scan")
    plt.xlabel(param_variable)
    plt.ylabel(r"$\chi^2$")
    plt.xlim((min(theta_scan),max(theta_scan)))
    plt.yscale('log')
    plt.ylim((min(chi2_values)/10, max(chi2_values)*10))
    plt.savefig(output_dir / "chi2_scan.pdf")


def chi2_heatmap_plot(
    theta_axes: list[np.ndarray],
    chi2_values: np.ndarray,
    parameter_names: list[str],
    output_dir: Path,
):
    if len(theta_axes) != 2:
        return

    grid = chi2_values.reshape(len(theta_axes[0]), len(theta_axes[1]))
    plt.clf()
    plt.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=[theta_axes[0][0], theta_axes[0][-1], theta_axes[1][0], theta_axes[1][-1]],
    )
    plt.colorbar(label=r"$\chi^2$")
    plt.xlabel(parameter_names[0])
    plt.ylabel(parameter_names[1])
    plt.title(r"$\chi^2$ heatmap")
    plt.savefig(output_dir / "chi2_scan_heatmap.pdf")


def review_plot(hypothesis_shape: np.ndarray, hypothesis_edges: np.ndarray,
                 best_infer_shape: np.ndarray, truth_infer_shape: np.ndarray, 
                 best_label: str, truth_label: str,
                 observable: str, output_dir: Path):
    plt.clf()
    plt.stairs(hypothesis_shape, hypothesis_edges, label="Hypothesis")
    plt.stairs(best_infer_shape, hypothesis_edges, label=f"Inference - best {best_label}")
    plt.stairs(truth_infer_shape, hypothesis_edges, label=f"Inference - truth {truth_label}")
    plt.title("Inference / Hypothesis shape comparison")
    plt.xlabel("%s [GeV]" % observable)
    plt.ylabel("Normalised events")
    plt.legend()
    plt.savefig(output_dir / "inference_review.pdf")


def _get_truth_parameter_point(cfg: DictConfig, datamodule) -> tuple[float, ...]:
    if hasattr(cfg.inference, "truth_parameters"):
        truth_parameters = cfg.inference.truth_parameters
        return tuple(float(truth_parameters[name]) for name in datamodule.parameter_names)

    if len(datamodule.parameter_names) == 1 and hasattr(cfg.inference, "truth_parameter"):
        return (float(cfg.inference.truth_parameter),)

    raise ValueError(
        "Multi-parameter inference requires inference.truth_parameters keyed by parameter name."
    )


def _build_scan_axes(cfg: DictConfig, datamodule) -> list[np.ndarray]:
    if hasattr(cfg.inference, "scan_parameters"):
        scan_specs = {str(spec.name): spec for spec in cfg.inference.scan_parameters}
        axes = []
        for parameter_name in datamodule.parameter_names:
            if parameter_name not in scan_specs:
                raise ValueError(f"Missing inference.scan_parameters entry for '{parameter_name}'")
            spec = scan_specs[parameter_name]
            axes.append(np.linspace(float(spec.min), float(spec.max), int(spec.n_points)))
        return axes

    if len(datamodule.parameter_names) == 1 and all(hasattr(cfg.inference, key) for key in ["theta_min", "theta_max", "n_points"]):
        return [np.linspace(cfg.inference.theta_min, cfg.inference.theta_max, cfg.inference.n_points)]

    logger.warning("No inference.scan_parameters provided; falling back to discrete training+holdout axes.")
    axes = []
    for parameter_name in datamodule.parameter_names:
        training_values = datamodule.parameter_axes.get(parameter_name, [])
        holdout_values = datamodule.holdout_parameter_axes.get(parameter_name, [])
        combined = sorted(set(training_values) | set(holdout_values))
        if len(combined) == 0:
            raise ValueError(f"Unable to infer scan axis for parameter '{parameter_name}'")
        axes.append(np.asarray(combined, dtype=float))
    return axes


def _scale_theta_point(theta_point: tuple[float, ...], datamodule) -> np.ndarray:
    scaled = []
    for theta_value, transformer_name in zip(theta_point, datamodule.parameter_transformer_names):
        transformer = datamodule.scaler.named_transformers_[transformer_name]
        scaled_value = transformer.transform(np.asarray([[theta_value]], dtype=float))[0, 0]
        scaled.append(float(scaled_value))
    return np.asarray(scaled, dtype=np.float32)


def _infer_shape_for_point(
    theta_point: tuple[float, ...],
    datamodule,
    x_ref_tensor: torch.Tensor,
    x_ref_inverted: np.ndarray,
    model_pp,
    model_pn,
    c_zero: float,
    c_one: float,
    hypothesis_shape: np.ndarray,
    rosmm_sign: float,
):
    theta_scaled = _scale_theta_point(theta_point, datamodule)
    theta_tensor = torch.tensor(theta_scaled, dtype=torch.float32)
    rew = rosmm(theta_tensor, x_ref_tensor, model_pp, model_pn, c_zero, c_one).detach().numpy()
    rew = rosmm_sign * rew / (np.abs(rew).sum() / np.abs(hypothesis_shape).sum())
    return _hist(x_ref_inverted, rew)

def run_inference(datamodule, model_class, cfg: DictConfig) -> None:
    if not hasattr(cfg, "inference"):
        raise ValueError("Missing inference section in config.")

    output_dir = Path(resolve_runtime_path(getattr(cfg.inference, "output_dir", "results/inference")))
    output_dir.mkdir(parents=True, exist_ok=True)

    observable = getattr(cfg.inference, "observable", None)
    if observable is None:
        logger.error("Inference config must specify an observable to scan.")
        raise ValueError("Missing observable in inference config.")
    logger.info(f"Running inference scan for observable: {observable}")

    logger.info("Loading models for inference...")
    pp_ckpt = cfg.inference.model_pp_checkpoint
    pn_ckpt = cfg.inference.model_pn_checkpoint

    logger.info("Setting datamodule...")
    datamodule.setup(stage="inference")

    model_pp = model_class(cfg)
    model_pn = model_class(cfg)

    model_pp = load_checkpoint_into_model(model_pp, get_latest_checkpoint_path(pp_ckpt)).model
    model_pn = load_checkpoint_into_model(model_pn, get_latest_checkpoint_path(pn_ckpt)).model

    model_pp.eval()
    model_pn.eval()

    x_indices = datamodule.x_column_indices
    weight_index = datamodule.weight_column_index

    logger.debug("Finding observable key in transformed features...")
    observable_key = None
    inverse_observable_transform = None
    for obs in datamodule.observables_config:
        group_name, variable_name = parse_feature_spec(obs)
        if variable_name == observable:
            observable_key = feature_key(group_name, variable_name)
            inverse_observable_transform = datamodule.scaler.named_transformers_[f"obs_{observable}"].inverse_transform
            break
    if observable_key is None:
        raise ValueError(f"Inference requires {observable} to be listed in dataset.observables.")
    observable_index = datamodule.transformed_feature_keys.index(observable_key)

    logger.info("Building reference background sample...")
    X_test = datamodule.test_dataset.tensors[0].cpu().numpy()
    y_test = datamodule.test_dataset.tensors[1].cpu().numpy()

    bg_mask = y_test[:, 0] == 0
    X_ref = X_test[bg_mask]
    if len(X_ref) == 0:
        raise RuntimeError("No background events available in the test split for inference.")
    X_ref_inverted = inverse_observable_transform(X_ref[:, observable_index].reshape(-1, 1)).flatten()

    if datamodule.X_holdout is None or len(datamodule.X_holdout) == 0:
        raise RuntimeError("No holdout events available for inference. Check holdout parameters in config.")

    X_hyp = datamodule.X_holdout
    y_hyp = datamodule.y_holdout

    truth_point = _get_truth_parameter_point(cfg, datamodule)
    if truth_point not in datamodule.parameter_point_to_category:
        raise ValueError(f"Truth parameter point {truth_point} is not present in the dataset.")
    holdout_category = datamodule.parameter_point_to_category[truth_point]
    signal_holdout = X_hyp[(y_hyp[:, 0] == 1) & (y_hyp[:, 1] == holdout_category)]
    if len(signal_holdout) == 0:
        raise RuntimeError(f"No holdout signal events found for truth point {truth_point}.")
    logger.info("Reversing transformation...")
    signal_holdout = signal_holdout.copy()
    signal_holdout[:, observable_index] = inverse_observable_transform(signal_holdout[:, observable_index].reshape(-1, 1)).flatten()


    logger.info("Generating hypotheis plot...")
    hypothesis_shape, hypothesis_var, hypothesis_edges = _hist(
        signal_holdout[:, observable_index], signal_holdout[:, weight_index]
    )
    hypothesis_plot(hypothesis_shape, hypothesis_var, hypothesis_edges, observable, output_dir)

    theta_axes = _build_scan_axes(cfg, datamodule)
    theta_scan = [tuple(float(value) for value in point) for point in itertools.product(*theta_axes)]
    chi2_values = []

    # C0/C1 are estimated from train split as in the notebook recipe.
    X_train = datamodule.train_dataset.tensors[0].cpu().numpy()
    y_train = datamodule.train_dataset.tensors[1].cpu().numpy()
    bg_train = y_train[:, 0] == 0
    sig_train = y_train[:, 0] == 1
    bg_train_pos = bg_train & (X_train[:, weight_index] >= 0)
    sig_train_pos = sig_train & (X_train[:, weight_index] >= 0)

    logger.debug("N BG %d", bg_train.sum())
    logger.debug("N SIG %d", sig_train.sum())
    logger.debug("N BG w>0 %d", bg_train_pos.sum())
    logger.debug("N SIG w>0 %d", sig_train_pos.sum())

    logger.info("Estimating RoSMM C0 and C1 from training data...")
    c_zero = float(
        X_train[bg_train_pos, weight_index].sum() / X_train[bg_train, weight_index].sum()
    )
    logger.info(f"Estimated C0: {c_zero:.4f}")
    c_one = float(
        X_train[sig_train_pos, weight_index].sum() / X_train[sig_train, weight_index].sum()
    )
    logger.info(f"Estimated C1: {c_one:.4f}")

    x_ref_tensor = torch.tensor(X_ref[:, x_indices], dtype=torch.float32)

    logger.info("Running inference scan...")
    counter = 1
    rosmm_sign = cfg.inference.rosmm_sign
    if (abs(rosmm_sign) - 1) > 0.0001:
        logger.error("Only +/- 1.0 values can be passed as RoSMM sign.")
        raise ValueError(f"{rosmm_sign} is not a valid value for RoSMM sign.")
    scan_rows = []
    for theta_point in theta_scan:
        infer_shape, infer_var, _ = _infer_shape_for_point(
            theta_point=theta_point,
            datamodule=datamodule,
            x_ref_tensor=x_ref_tensor,
            x_ref_inverted=X_ref_inverted,
            model_pp=model_pp,
            model_pn=model_pn,
            c_zero=c_zero,
            c_one=c_one,
            hypothesis_shape=hypothesis_shape,
            rosmm_sign=rosmm_sign,
        )
        theta_label = parameter_point_label(datamodule.parameter_names, theta_point)
        if len(datamodule.parameter_names) == 1:
            inference_scan_plot(infer_shape, hypothesis_shape, hypothesis_edges, observable, theta_label, output_dir, counter)
        chi2_value = chi_squared(hypothesis_shape, infer_shape, infer_var)
        chi2_values.append(chi2_value)
        scan_row = {name: value for name, value in zip(datamodule.parameter_names, theta_point)}
        scan_row["chi2"] = chi2_value
        scan_rows.append(scan_row)
        counter += 1

    chi2_values_array = np.asarray(chi2_values, dtype=float)
    pd.DataFrame(scan_rows).to_csv(output_dir / "chi2_scan.csv", index=False)

    best_idx = int(np.argmin(chi2_values))
    best_theta = theta_scan[best_idx]
    logger.info("Best parameter point: %s, chi2=%.4f", best_theta, chi2_values[best_idx])

    if len(datamodule.parameter_names) == 1:
        logger.info("Generating chi2 plot...")
        chi2_plot(theta_axes[0], chi2_values_array, datamodule.parameter_names[0], output_dir)
    elif len(datamodule.parameter_names) == 2:
        logger.info("Generating chi2 heatmap...")
        chi2_heatmap_plot(theta_axes, chi2_values_array, datamodule.parameter_names, output_dir)

    logger.info("Generating review plot...")
    logger.debug("Best parameter point inference...")
    best_infer_shape, _, _ = _infer_shape_for_point(
        theta_point=best_theta,
        datamodule=datamodule,
        x_ref_tensor=x_ref_tensor,
        x_ref_inverted=X_ref_inverted,
        model_pp=model_pp,
        model_pn=model_pn,
        c_zero=c_zero,
        c_one=c_one,
        hypothesis_shape=hypothesis_shape,
        rosmm_sign=rosmm_sign,
    )
    logger.debug("Truth parameter point inference...")
    truth_infer_shape, _, _ = _infer_shape_for_point(
        theta_point=truth_point,
        datamodule=datamodule,
        x_ref_tensor=x_ref_tensor,
        x_ref_inverted=X_ref_inverted,
        model_pp=model_pp,
        model_pn=model_pn,
        c_zero=c_zero,
        c_one=c_one,
        hypothesis_shape=hypothesis_shape,
        rosmm_sign=rosmm_sign,
    )
    logger.debug("Plotting...")
    review_plot(
        hypothesis_shape,
        hypothesis_edges,
        best_infer_shape,
        truth_infer_shape,
        parameter_point_label(datamodule.parameter_names, best_theta),
        parameter_point_label(datamodule.parameter_names, truth_point),
        observable,
        output_dir,
    )
    

