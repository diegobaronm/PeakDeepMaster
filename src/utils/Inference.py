import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
import logging
import matplotlib.pyplot as plt

from src.data.DataHelpers import feature_key, parse_feature_spec
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

def inference_scan_plot(infer_shape: np.ndarray, hypothesis_shape: np.ndarray, hypothesis_edges: np.ndarray, observable: str, param_variable: str, theta: float, output_dir: Path, postfix: int):
    plt.clf()
    plt.stairs(infer_shape, hypothesis_edges, label="Inference")
    plt.stairs(hypothesis_shape, hypothesis_edges, label="Hypothesis")
    plt.title("Inference shape %s = %s" % (param_variable, str(theta)))
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

def review_plot(hypothesis_shape: np.ndarray, hypothesis_edges: np.ndarray,
                 best_infer_shape: np.ndarray, truth_infer_shape: np.ndarray, 
                 param_variable: str, best_theta: float, truth_theta: float,
                 observable: str, output_dir: Path):
    plt.clf()
    plt.stairs(hypothesis_shape, hypothesis_edges, label="Hypothesis")
    plt.stairs(best_infer_shape, hypothesis_edges, label="Inference - best %s = %.4f" % (param_variable, best_theta))
    plt.stairs(truth_infer_shape, hypothesis_edges, label="Inference - truth %s = %.4f" % (param_variable, truth_theta))
    plt.title("Inference / Hypothesis shape comparison")
    plt.xlabel("%s [GeV]" % observable)
    plt.ylabel("Normalised events")
    plt.legend()
    plt.savefig(output_dir / "inference_review.pdf")

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
    parameter_index = datamodule.parameter_column_index
    weight_index = datamodule.weight_column_index

    logger.debug("Finding observable key in transformed features...")
    observable_key = None
    inverse_observable_transform = None
    param_group, param_variable = parse_feature_spec(datamodule.parameter_spec)
    parameter_transformer_name = f"param_{param_variable}"
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
    # Build a reference background sample from test split at high transformed parameter.
    X_test = datamodule.test_dataset.tensors[0].cpu().numpy()
    y_test = datamodule.test_dataset.tensors[1].cpu().numpy()

    bg_mask = (y_test[:, 0] == 0) & (X_test[:, parameter_index] >= 0.98)
    X_ref = X_test[bg_mask]
    X_ref_inverted = inverse_observable_transform(X_ref[:, observable_index].reshape(-1, 1)).flatten()

    # Pick first holdout parameter as hypothesis if available.
    if datamodule.X_holdout is None or len(datamodule.X_holdout) == 0:
        raise RuntimeError("No holdout events available for inference. Check holdout parameters in config.")

    X_hyp = datamodule.X_holdout
    y_hyp = datamodule.y_holdout

    holdout_category =  datamodule.parameters_category_dict[cfg.inference.truth_parameter]
    signal_holdout = X_hyp[y_hyp[:, 0] == 1 & (y_hyp[:, 1] == holdout_category)]
    logger.info("Reversing transformation...")
    signal_holdout[:, observable_index] = inverse_observable_transform(signal_holdout[:, observable_index].reshape(-1, 1)).flatten()


    logger.info("Generating hypotheis plot...")
    hypothesis_shape, hypothesis_var, hypothesis_edges = _hist(
        signal_holdout[:, observable_index], signal_holdout[:, weight_index]
    )
    hypothesis_plot(hypothesis_shape, hypothesis_var, hypothesis_edges, observable, output_dir)

    theta_scan = np.linspace(cfg.inference.theta_min, cfg.inference.theta_max, cfg.inference.n_points)
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
        log.error("Only +/- 1.0 values can be passed as RoSMM sign.")
        raise ValueError(f"{rosmm_sign} is not a valid value for RoSMM sign.")
    for theta in theta_scan:
        theta_scaled = datamodule.scaler.named_transformers_[parameter_transformer_name].transform([[theta]])[0, 0]
        theta_tensor = torch.tensor([theta_scaled], dtype=torch.float32)

        rew = rosmm(theta_tensor, x_ref_tensor, model_pp, model_pn, c_zero, c_one).detach().numpy()
        rew = rosmm_sign * rew / (np.abs(rew).sum() / np.abs(hypothesis_shape).sum())

        infer_shape, infer_var, _ = _hist(X_ref_inverted, rew)
        inference_scan_plot(infer_shape, hypothesis_shape, hypothesis_edges, observable, param_variable, theta, output_dir, counter)
        logger.debug("Calculating chi2 for theta = %.4f", theta)
        chi2_values.append(chi_squared(hypothesis_shape, infer_shape, infer_var))
        counter+=1

    best_idx = int(np.argmin(chi2_values))
    logger.info(f"Best theta: {theta_scan[best_idx]:.4f}, chi2={chi2_values[best_idx]:.4f}")

    logger.info("Generating chi2 plot...")
    chi2_plot(theta_scan, chi2_values, param_variable, output_dir)

    logger.info("Generating review plot...")
    logger.debug("Best theta inference...")
    theta_scaled = datamodule.scaler.named_transformers_[parameter_transformer_name].transform([[theta_scan[best_idx]]])[0, 0]
    theta_tensor = torch.tensor([theta_scaled], dtype=torch.float32)
    rew = rosmm(theta_tensor, x_ref_tensor, model_pp, model_pn, c_zero, c_one).detach().numpy()
    rew = rosmm_sign * rew / (np.abs(rew).sum() / np.abs(hypothesis_shape).sum())
    best_infer_shape, best_infer_var, _ = _hist(X_ref_inverted, rew)
    logger.debug("Truth theta inference...")
    theta_scaled = datamodule.scaler.named_transformers_[parameter_transformer_name].transform([[cfg.inference.truth_parameter]])[0, 0]
    theta_tensor = torch.tensor([theta_scaled], dtype=torch.float32)
    rew = rosmm(theta_tensor, x_ref_tensor, model_pp, model_pn, c_zero, c_one).detach().numpy()
    rew = rosmm_sign * rew / (np.abs(rew).sum() / np.abs(hypothesis_shape).sum())
    truth_infer_shape, truth_infer_var, _ = _hist(X_ref_inverted, rew)
    logger.debug("Plotting...")
    review_plot(hypothesis_shape, hypothesis_edges, best_infer_shape, truth_infer_shape, param_variable, theta_scan[best_idx], cfg.inference.truth_parameter, observable, output_dir)
    

