import numpy as np
import torch
from omegaconf import DictConfig

from src.data.DataHelpers import feature_key, parse_feature_spec
from src.utils.utils import get_latest_checkpoint_path, load_checkpoint_into_model


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
    denom = np.where(expected_variance <= 0, 1e-8, expected_variance)
    return float(np.sum(((observed - expected) ** 2) / denom))


def _hist(values: np.ndarray, weights: np.ndarray, n_bins: int = 50, x_min: float = 500.0, x_max: float = 1200.0):
    hist, edges = np.histogram(values, bins=n_bins, range=(x_min, x_max), weights=weights)
    var_hist, _ = np.histogram(values, bins=n_bins, range=(x_min, x_max), weights=weights ** 2)
    return hist, var_hist, edges


def run_inference(datamodule, model_class, cfg: DictConfig) -> None:
    if not hasattr(cfg, "inference"):
        raise ValueError("Missing inference section in config.")

    pp_ckpt = cfg.inference.model_pp_checkpoint
    pn_ckpt = cfg.inference.model_pn_checkpoint

    datamodule.setup(stage="infer")

    model_pp = model_class(cfg)
    model_pn = model_class(cfg)

    model_pp = load_checkpoint_into_model(model_pp, get_latest_checkpoint_path(pp_ckpt)).model
    model_pn = load_checkpoint_into_model(model_pn, get_latest_checkpoint_path(pn_ckpt)).model

    model_pp.eval()
    model_pn.eval()

    x_indices = datamodule.x_column_indices
    coupling_index = datamodule.coupling_column_index
    weight_index = datamodule.weight_column_index

    mttbar_key = None
    param_group, param_variable = parse_feature_spec(datamodule.parameter_spec)
    coupling_transformer_name = f"param_{param_variable}"
    for obs in datamodule.observables_config:
        group_name, variable_name = parse_feature_spec(obs)
        if variable_name == "mttbar":
            mttbar_key = feature_key(group_name, variable_name)
            break
    if mttbar_key is None:
        raise ValueError("Inference requires mttbar to be listed in dataset.observables.")
    mttbar_index = datamodule.transformed_feature_keys.index(mttbar_key)

    # Build a reference background sample from test split at high transformed coupling.
    X_test = datamodule.test_dataset.tensors[0].cpu().numpy()
    y_test = datamodule.test_dataset.tensors[1].cpu().numpy()

    bg_mask = (y_test[:, 0] == 0) & (X_test[:, coupling_index] >= 0.98)
    X_ref = X_test[bg_mask]

    # Pick first holdout coupling as hypothesis if available.
    if datamodule.X_holdout is None or len(datamodule.X_holdout) == 0:
        raise RuntimeError("No holdout events available for inference. Check holdout couplings in config.")

    X_hyp = datamodule.X_holdout
    y_hyp = datamodule.y_holdout
    signal_holdout = X_hyp[y_hyp[:, 0] == 1]

    hypothesis_shape, hypothesis_var, _ = _hist(
        signal_holdout[:, mttbar_index], np.abs(signal_holdout[:, weight_index])
    )

    theta_scan = np.linspace(cfg.inference.theta_min, cfg.inference.theta_max, cfg.inference.n_points)
    chi2_values = []

    # C0/C1 are estimated from train split as in the notebook recipe.
    X_train = datamodule.train_dataset.tensors[0].cpu().numpy()
    y_train = datamodule.train_dataset.tensors[1].cpu().numpy()
    bg_train = y_train[:, 0] == 0
    sig_train = y_train[:, 0] == 1

    c_zero = float(
        X_train[bg_train & (X_train[:, weight_index] >= 0), weight_index].sum() / X_train[bg_train, weight_index].sum()
    )
    c_one = float(
        X_train[sig_train & (X_train[:, weight_index] >= 0), weight_index].sum() / X_train[sig_train, weight_index].sum()
    )

    x_ref_tensor = torch.tensor(X_ref[:, x_indices], dtype=torch.float32)

    for theta in theta_scan:
        theta_scaled = datamodule.scaler.named_transformers_[coupling_transformer_name].transform([[theta]])[0, 0]
        theta_tensor = torch.tensor([theta_scaled], dtype=torch.float32)

        rew = -1.0 * rosmm(theta_tensor, x_ref_tensor, model_pp, model_pn, c_zero, c_one).detach().numpy()
        rew = rew / (np.abs(rew).sum() / max(np.abs(hypothesis_shape).sum(), 1e-8))

        infer_shape, infer_var, _ = _hist(X_ref[:, mttbar_index], rew)
        chi2_values.append(chi_squared(hypothesis_shape, infer_shape, infer_var))

    best_idx = int(np.argmin(chi2_values))
    print(f"Best theta: {theta_scan[best_idx]:.4f}, chi2={chi2_values[best_idx]:.4f}")
