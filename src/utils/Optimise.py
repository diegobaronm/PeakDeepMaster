import logging
from typing import Any

import optuna
from optuna_integration import PyTorchLightningPruningCallback
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import EarlyStopping

from src.data.DataModule import PeakDeepMasterDataModule
from src.models.RatioEstimator import LLHRatioEstimator
from src.utils.utils import set_execution_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter sampling
# ---------------------------------------------------------------------------

def _suggest_hyperparameter(trial: optuna.trial.Trial, hp_spec: dict) -> Any:
    """Suggest a single hyperparameter value from an Optuna *trial* according
    to a specification dictionary parsed from the YAML config.

    Each element in ``optimise.hyper_parameters`` is a dict with exactly one
    "name" key (the first key that is *not* ``type``, ``range``, ``values``,
    or ``log``) whose value is ``None``, plus the metadata keys.

    Supported types
    ---------------
    * ``int``         – ``trial.suggest_int(name, low, high, log=…)``
    * ``float``       – ``trial.suggest_float(name, low, high, log=…)``
    * ``categorical`` – ``trial.suggest_categorical(name, choices)``
    """
    meta_keys = {"type", "range", "values", "log"}
    # Discover the parameter name – the first key not in meta_keys
    name = None
    for key in hp_spec:
        if key not in meta_keys:
            name = key
            break
    if name is None:
        raise ValueError(f"Could not determine parameter name from spec: {hp_spec}")

    hp_type = str(hp_spec["type"]).lower()
    use_log = bool(hp_spec.get("log", False))

    if hp_type == "int":
        low, high = hp_spec["range"]
        return trial.suggest_int(name, int(low), int(high), log=use_log)
    elif hp_type == "float":
        low, high = hp_spec["range"]
        return trial.suggest_float(name, float(low), float(high), log=use_log)
    elif hp_type == "categorical":
        choices = list(hp_spec["values"])
        return trial.suggest_categorical(name, choices)
    else:
        raise ValueError(f"Unsupported hyperparameter type '{hp_type}' for '{name}'")


def _suggest_all_hyperparameters(trial: optuna.trial.Trial, hp_specs: list) -> dict:
    """Return ``{name: value}`` for every hyper-parameter specification."""
    params: dict[str, Any] = {}
    for spec in hp_specs:
        meta_keys = {"type", "range", "values", "log"}
        name = next((k for k in spec if k not in meta_keys), None)
        if name is None:
            continue
        params[name] = _suggest_hyperparameter(trial, spec)
    return params


# ---------------------------------------------------------------------------
# Config override helpers
# ---------------------------------------------------------------------------

def _get_hp_to_cfg_path(cfg: DictConfig) -> dict[str, tuple[str, ...]]:
    """Return configuration paths supported by this project's MLP."""
    return {
        "hidden_dim": ("model", "hidden_dim"),
        "hidden_layers": ("model", "hidden_layers"),
        "dropout": ("model", "dropout"),
        "learning_rate": ("train", "learning_rate"),
        "weight_decay": ("train", "weight_decay"),
        "batch_size": ("dataset", "train", "batch_size"),
    }


def _apply_hyperparameters(cfg: DictConfig, params: dict) -> DictConfig:
    """Return a *mutable copy* of *cfg* with the sampled hyper-parameters
    written into the correct config sections."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # plain dict
    cfg = OmegaConf.create(cfg)                       # mutable DictConfig
    OmegaConf.set_struct(cfg, False)                  # allow new keys

    hp_to_cfg_path = _get_hp_to_cfg_path(cfg)

    for hp_name, value in params.items():
        if hp_name in hp_to_cfg_path:
            path = hp_to_cfg_path[hp_name]
            target = cfg
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        else:
            logger.warning("No config mapping for hyperparameter '%s'; skipping.", hp_name)

    # If batch_size was sampled, keep validation batches aligned.
    if "batch_size" in params:
        cfg.dataset.val.batch_size = params["batch_size"]

    return cfg


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    """Single Optuna trial: sample hyper-parameters, build model & data,
    train, and return the best validation loss.

    Training budget and early-stopping patience can be overridden via
    ``cfg.optimise.max_epochs`` and ``cfg.optimise.early_stopping_patience``
    to keep HPO trials shorter than full training runs."""


    # 1. Sample hyper-parameters from the search space
    hp_config = cfg.optimise.get("hyper_parameters", None)
    hp_specs = [] if hp_config is None else OmegaConf.to_container(hp_config, resolve=True)
    if not hp_specs:
        raise ValueError(
            "optimise.hyper_parameters is empty; uncomment at least one search specification."
        )
    params = _suggest_all_hyperparameters(trial, hp_specs)
    logger.info("Trial %d – sampled parameters: %s", trial.number, params)

    # 2. Build a trial-specific config with the sampled values
    trial_cfg = _apply_hyperparameters(cfg, params)

    # 3. Build the local data module and model.
    datamodule = PeakDeepMasterDataModule(trial_cfg)
    data_pct = float(trial_cfg.optimise.get("data_percentage", 1.0))
    if not 0 < data_pct <= 1:
        raise ValueError("optimise.data_percentage must be in the range (0, 1].")
    datamodule.data_percentage = data_pct
    model = LLHRatioEstimator(trial_cfg)

    # Compile only when explicitly requested by the trial configuration.
    if bool(trial_cfg.train.get("compile", False)):
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    # 6. Configure the Lightning Trainer with pruning callback
    device = set_execution_device(trial_cfg.general.device)
    monitor_metric = str(trial_cfg.train.get("monitor_metric", "val_loss"))
    monitor_mode = str(trial_cfg.train.get("monitor_mode", "min"))

    # Use optimise-specific overrides for max_epochs and early stopping
    optim_max_epochs = int(cfg.optimise.get("max_epochs", trial_cfg.train.n_epochs))
    optim_es_patience = int(cfg.optimise.get(
        "early_stopping_patience",
        trial_cfg.train.get("early_stopping_patience",
                            trial_cfg.train.get("lr_patience", 15)),
    ))

    callbacks = [
        PyTorchLightningPruningCallback(trial, monitor=monitor_metric),
        EarlyStopping(
            monitor=monitor_metric,
            patience=optim_es_patience,
            mode=monitor_mode,
            verbose=False,
            strict=False,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=optim_max_epochs,
        accelerator=device,
        devices="auto",
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,            # no TensorBoard logging for trials
    )

    # 7. Train
    torch.set_float32_matmul_precision("high")
    trainer.fit(model=model, datamodule=datamodule)

    # 8. Return the best validation metric
    val_loss = trainer.callback_metrics.get(monitor_metric)
    if val_loss is None:
        raise optuna.TrialPruned("No validation metric recorded.")
    return float(val_loss)


# ---------------------------------------------------------------------------
# Pruner factory
# ---------------------------------------------------------------------------

def _build_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    """Instantiate an Optuna pruner from the config ``optimise.pruner`` key."""
    pruner_name = str(cfg.optimise.get("pruner", "MedianPruner"))
    pruner_map = {
        "medianpruner":      optuna.pruners.MedianPruner,
        "percentile":        optuna.pruners.PercentilePruner,
        "hyperband":         optuna.pruners.HyperbandPruner,
        "nop":               optuna.pruners.NopPruner,
        "threshold":         optuna.pruners.ThresholdPruner,
    }
    cls = pruner_map.get(pruner_name.lower())
    if cls is None:
        logger.warning("Unknown pruner '%s'; falling back to MedianPruner.", pruner_name)
        cls = optuna.pruners.MedianPruner
    if cls is optuna.pruners.MedianPruner:
        return cls(
            n_startup_trials=int(cfg.optimise.get("n_startup_trials", 15)),
            n_warmup_steps=int(cfg.optimise.get("n_warmup_steps", 5)),
        )
    return cls()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_optimise(cfg: DictConfig) -> None:
    """Run Optuna hyperparameter optimisation as configured in
    ``cfg.optimise``."""
    set_seed(cfg.general.seed)

    optimise_cfg = cfg.optimise
    hp_config = optimise_cfg.get("hyper_parameters", None)
    hp_specs = [] if hp_config is None else OmegaConf.to_container(hp_config, resolve=True)
    if not hp_specs:
        raise ValueError(
            "optimise.hyper_parameters is empty; uncomment at least one search specification."
        )
    n_trials = int(optimise_cfg.get("n_trials", 50))
    timeout = optimise_cfg.get("timeout", None)
    if timeout is not None:
        timeout = float(timeout)

    pruner = _build_pruner(cfg)

    monitor_metric = str(cfg.train.get("monitor_metric", "val_loss"))
    monitor_mode = str(cfg.train.get("monitor_mode", "min"))
    direction = "minimize" if monitor_mode == "min" else "maximize"

    strategy = str(
        optimise_cfg.get(
            "optmiser_strategy",
            optimise_cfg.get("optimiser_strategy", "default"),
        )
    ).lower()
    if strategy == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.general.seed)
    elif strategy in {"default", "tpe"}:
        sampler = optuna.samplers.TPESampler(
            seed=cfg.general.seed,
            multivariate=True,
            n_startup_trials=int(optimise_cfg.get("n_startup_trials", 15)),
        )
    else:
        raise ValueError(f"Unsupported optimise strategy: {strategy}")

    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        sampler=sampler,
        study_name=f"optimise_{cfg.logging.train.experiment_name}",
    )

    logger.info("Setting num_workers to zero to avoid multiprocessing issues.")
    cfg.dataset.train.num_workers = 0
    cfg.dataset.val.num_workers = 0

    logger.info(
        "Starting Optuna study: n_trials=%d, timeout=%s, direction=%s, pruner=%s",
        n_trials, timeout, direction, type(pruner).__name__,
    )

    study.optimize(
        lambda trial: _objective(trial, cfg),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=1,
    )

    # ---- Report results ----
    logger.info("=" * 60)
    logger.info("Optimisation complete.")
    logger.info("  Number of finished trials: %d", len(study.trials))

    best = study.best_trial
    logger.info("  Best trial (#%d):", best.number)
    logger.info("    Value (%s): %.6f", monitor_metric, best.value)
    logger.info("    Parameters:")
    for key, value in best.params.items():
        logger.info("      %s: %s", key, value)
    logger.info("=" * 60)