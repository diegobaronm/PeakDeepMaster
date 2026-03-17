# PeakDeepMaster

Simulation-based inference (SBI) for heavy Higgs events with strong interference with the Standard Model. The framework trains likelihood-ratio estimators using PyTorch Lightning and performs parameter estimation via the RoSMM (Ratio of Signed Morphing Models) method, supporting both 1D and 2D parameter scans with pseudo-experiment uncertainty estimation.

## Setup

```bash
conda env create -f requirements.yaml
conda activate peak_deep_master
pip install -e .
```

Requires Python >= 3.10.

## Modes

The runner script uses Hydra for configuration management. All output is written under an `outputs/<date>/<time>/` directory created automatically by Hydra.

```bash
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=train
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=predict
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=performance
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=inference

python PeakDeepMaster.py --config-name config_1D_mass general.mode=train
python PeakDeepMaster.py --config-name config_2D general.mode=train
```

### Available modes

| Mode          | Description |
|---------------|-------------|
| `train`       | Fit a new `LLHRatioEstimator` model and save checkpoints. |
| `predict`     | Run prediction on the test split and export a CSV with scores, labels, and categories. |
| `performance` | Evaluate test and holdout splits, producing ROC curves, score distributions, and summary CSVs. |
| `inference`   | Scan over parameter values using the RoSMM method and compute an $L^2$ statistic to find the best-fit parameter(s). |

## Project Structure

```
PeakDeepMaster.py          # Hydra entry point — dispatches to the selected mode.
configs/
  config_1D_coupling.yaml  # 1D inference on coupling strength.
  config_1D_mass.yaml      # 1D inference on heavy Higgs mass.
  config_2D.yaml           # Joint 2D inference on coupling × mass.
src/
  data/
    DataHelpers.py          # H5 reading, parameter grid building, data augmentation, weight normalisation.
    DataModule.py           # PeakDeepMasterDataModule (Lightning DataModule) — loading, splitting, scaling.
    DataScaler.py           # Feature scaling (LogScaler, StandardScaler, MinMaxScaler) via ColumnTransformer.
  models/
    RatioEstimator.py       # RatioEstimatorNet (MLP) and LLHRatioEstimator (Lightning Module).
  utils/
    Train.py                # Training loop with EarlyStopping, ModelCheckpoint, and TensorBoard logging.
    Predict.py              # Predict mode — runs model on test split, writes CSV.
    Performance.py          # Performance mode — score distributions and ROC curves per parameter point.
    Inference.py            # Inference mode — RoSMM likelihood-ratio reweighting and chi-squared scanning.
    PseudoExperiments.py    # Pseudo-experiment-based parameter uncertainty estimation.
    utils.py                # Logging setup, seed, device selection, checkpoint loading, path resolution.
tests/
  configs/test_config.yaml  # Lightweight test configuration (CPU, small model, few epochs).
  data/                     # Test data placeholder.
```

## Configuration Layout

The configs in `configs/` share the same top-level structure:

- `general`: run mode and global runtime settings.
- `logging`: console/file/TensorBoard logging behavior.
- `dataset`: input file, feature definitions, parameter grids, weights, and dataloader settings.
- `model`: network architecture.
- `train`: optimizer and training controls.
- `predict`: checkpoint and CSV output settings.
- `performance`: checkpoint and plot output settings.
- `inference`: RoSMM inference workflow settings.

### `general`

| Key      | Description |
|----------|-------------|
| `mode`   | One of `train`, `predict`, `performance`, `inference`. |
| `device` | Passed to Lightning (`cuda`, `cpu`, `mps`). Falls back to CPU if unavailable. |
| `seed`   | Seed for reproducible splitting, sampling, and training. |

### `logging`

| Key                      | Description |
|--------------------------|-------------|
| `level`                  | Log level (`DEBUG`, `INFO`, `WARNING`, etc.). |
| `format`, `datefmt`      | Standard Python logging format strings. |
| `colors`                 | Colored console output (`auto`, `true`, `false`). Respects `NO_COLOR` env var. |
| `file`                   | Optional log file path; `null` keeps logging on the console only. |
| `capture_warnings`       | Redirects Python warnings into the logger. |
| `events_to_log_in_debug` | List of event indices to log during data preparation (for debugging). |
| `train.output_dir`       | TensorBoard log directory (used during training). |
| `train.experiment_name`  | TensorBoard experiment name. |

### `dataset`

This is the most important section because it defines both the H5 input and how columns are mapped into the model.

| Key                        | Description |
|----------------------------|-------------|
| `input_h5_path`            | H5 file to read. Relative paths are resolved from the project root. |
| `max_events_per_parameter` | Cap used when subsampling events per parameter point (`-1` = no limit). |
| `observables`              | Ordered list of input features used as `x` by the classifier. |
| `parameters`               | One or more SBI parameters (supports multi-parameter / 2D inference). |
| `weights`                  | Exactly one event-weight entry. |
| `random_seed`              | Optional override for subsampling seed (defaults to `general.seed`). |
| `train` / `val`            | Dataloader `num_workers` and `batch_size`. |

Feature entries are written as YAML mappings:

```yaml
- EVENT_PARTICLES: top1_pt
  transformation: LogScaler
```

The non-`transformation` key identifies the H5 location as `INPUTS/GROUP/variable`. Supported transformations:

| Transformation   | Description |
|------------------|-------------|
| `LogScaler`      | Applies `log(x + offset)` (offset defaults to 1e-3). |
| `StandardScaler` | Scikit-learn `StandardScaler` (zero mean, unit variance). |
| `MinMaxScaler`   | Scikit-learn `MinMaxScaler` (scales to [0, 1]). |

The order of `dataset.observables` matters — it defines the model input vector, and `len(dataset.observables)` sets the network input dimension. Parameters default to `MinMaxScaler`, and weights are passed through unscaled. But they are later scaled in the input data creation.

#### Parameters

Each parameter entry controls both the model conditioning and the train/holdout split:

| Key                    | Description |
|------------------------|-------------|
| `values_for_training`  | Parameter values kept in the train/val/test splits. Required. |
| `values_for_testing`   | Holdout parameter values removed from training, used for evaluation and inference. |
| `transformation`       | Scaler applied to the parameter column (default: `MinMaxScaler`). |

**Multiple parameters are supported.** The `config_2D.yaml` demonstrates joint inference over `coupling` and `hmass`, where the training grid is the Cartesian product of both parameter axes.

Examples across configs:

| Config               | Parameter(s)          | Training values                                  | Holdout values  |
|----------------------|-----------------------|--------------------------------------------------|-----------------|
| `config_1D_coupling` | `coupling`            | `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]`    | `[0.5, 1.1]`   |
| `config_1D_mass`     | `hmass`               | `[600, 700, 800, 900, 1000]`                     | `[650, 870]`    |
| `config_2D`          | `coupling` × `hmass`  | `[0.2, …, 1.6]` × `[600, …, 1000]`             | `[0.5, 1.1]` × `[650, 870]` |

#### Background augmentation

Background events are augmented across all training parameter points. For each background event, copies are created with every training parameter value, so the background learns to be parameter-independent. Weights are normalised per category and sign before training.

### `model`

Configures the `RatioEstimatorNet`, a 4-layer MLP:

```
Linear(x_dim + theta_dim, hidden_dim) → ReLU
→ Linear(hidden_dim, hidden_dim) → ReLU → Dropout
→ Linear(hidden_dim, hidden_dim // 2) → ReLU → Dropout
→ Linear(hidden_dim // 2, 1)
```

| Key          | Description |
|--------------|-------------|
| `hidden_dim` | Width of the first hidden layers (second hidden layer is `hidden_dim // 2`). |
| `dropout`    | Dropout probability applied after the second and third layers. |

The input dimension is `len(observables) + len(parameters)` (`x_dim + theta_dim`). The output is a single logit passed through `BCEWithLogitsLoss` (weighted by absolute event weights).

### `train`

Consumed only when `general.mode=train`:

| Key                      | Description |
|--------------------------|-------------|
| `n_epochs`               | Maximum number of training epochs. |
| `learning_rate`          | AdamW learning rate. |
| `weight_decay`           | AdamW weight decay. |
| `lr_patience`            | `ReduceLROnPlateau` patience (epochs without improvement). |
| `lr_factor`              | `ReduceLROnPlateau` multiplicative factor. |
| `signal_weight_sign`     | Keep signal events with `positive` or `negative` weights. |
| `background_weight_sign` | Keep background events with `positive` or `negative` weights. |
| `compile`                | Optionally enable `torch.compile`. |

Training uses `EarlyStopping` (patience=15 on `val_loss`) and `ModelCheckpoint` (saves best model by `val_loss`). Logs are written via TensorBoard to the directory specified in `logging.train`.

### `predict`

Used when `general.mode=predict`:

| Key               | Description |
|-------------------|-------------|
| `checkpoint_path` | Directory containing a Lightning `.ckpt` file (the single checkpoint is loaded automatically). |
| `output_file`     | CSV file written with `prediction`, `label`, `category`, and `parameter_point` columns. |

### `performance`

Used when `general.mode=performance`:

| Key               | Description |
|-------------------|-------------|
| `checkpoint_path` | Checkpoint directory to evaluate. |
| `output_dir`      | Destination for ROC curves, score histograms, and summary CSV files. |
| `batch_size`      | Batch size used for evaluation. |

This mode evaluates both the regular **test** split and the **holdout** split. For each split it produces:

- Per-category score distributions (combined and individual plots).
- Per-category ROC curves with AUC values (combined and individual plots).
- A `roc_summary_<split>.csv` with AUC, signal count, and background count per category.

### `inference`

Used when `general.mode=inference`. This implements the RoSMM inference procedure:

1. Two pre-trained models are loaded: one trained on positive-weight signal (`model_pp`) and one on negative-weight signal (`model_pn`).
2. Constants $c_0$ and $c_1$ are estimated from the training data as the fraction of positive weights in background and signal respectively.
3. A reference background sample is built from the test split.
4. A hypothesis histogram is built from holdout signal events at the truth parameter point.
5. For each scan point $\theta$, the reference sample is reweighted using the RoSMM formula: $\text{RoSMM}(\theta) = r_{PP}(\theta) \cdot \frac{c_1}{c_0} + r_{PN}(\theta) \cdot \frac{1 - c_1}{c_0}$
6. The reweighted distribution is compared to the hypothesis via an $L^2$ statistic.
7. The parameter(s) minimising the $L^2$ are reported as the best fit.

#### 1D inference settings

| Key               | Description |
|-------------------|-------------|
| `truth_parameter` | The true parameter value (for reference on plots). |
| `theta_min`       | Lower bound of the parameter scan. |
| `theta_max`       | Upper bound of the parameter scan. |
| `n_points`        | Number of scan points. |

#### 2D / multi-parameter inference settings

| Key                  | Description |
|----------------------|-------------|
| `truth_parameters`   | Mapping of parameter name → true value (e.g., `{coupling: 0.5, hmass: 870.0}`). |
| `scan_parameters`    | List of `{name, min, max, n_points}` entries defining each scan axis. The scan grid is the Cartesian product. |

#### Common inference settings

| Key                             | Description |
|---------------------------------|-------------|
| `model_pp_checkpoint`           | Checkpoint for the positive-positive model. |
| `model_pn_checkpoint`           | Checkpoint for the positive-negative model. |
| `observable`                    | Observable used for the histogram comparison (must be in `dataset.observables`). |
| `rosmm_sign`                    | Sign multiplier for the RoSMM output (`+1.0` or `-1.0`). |
| `output_dir`                    | Result directory for inference outputs. |
| `n_pseudo_experiments`          | Number of pseudo-experiments for uncertainty estimation (0 to disable). |
| `pseudo_experiment_confidence`  | Confidence level for the pseudo-experiment interval (default: 0.95). |

#### Inference outputs

- `chi2_scan.csv`: $L^2$ value for each scanned parameter point.
- `hypothesis_shape.pdf`: the hypothesis histogram from holdout signal events.
- `inference_scan_<N>.pdf` (1D only): reweighted vs hypothesis comparison per scan point.
- `chi2_scan.pdf` (1D) or `chi2_scan_heatmap.pdf` (2D): $L^2$ scan visualisation with best-fit and truth markers.
- `inference_review.pdf`: best-fit and truth reweighted shapes overlaid on the hypothesis.
- When pseudo-experiments are enabled:
  - `pseudo_experiments.npy`: the generated pseudo-experiment histograms.
  - `inference_shapes.csv`: all scan-point inference shapes and uncertainties.
  - `best_fit_parameters.csv`: best-fit parameter and $\chi^2$ for each pseudo-experiment.
  - `pseudo_experiment_<param>.pdf`: histogram of best-fit values with confidence interval.

## Data Pipeline

1. **Loading**: Events are read from an H5 file structured as `INPUTS/<GROUP>/<variable>` with labels in `LABELS/CLASS`.
2. **Subsampling**: Up to `max_events_per_parameter` events are kept per unique parameter point.
3. **Structuring**: Observables, parameters, and weights are assembled into a feature matrix `X` and label matrix `y = [class, category]`.
4. **Weight normalisation** (train mode only): Weights are normalised per category and sign (positive/negative separately).
5. **Background augmentation**: Background events are replicated across all training parameter points.
6. **Feature scaling**: A `ColumnTransformer` applies the configured transformation to each column.
7. **Holdout removal**: Events matching `values_for_testing` are separated into a holdout set.
8. **Splitting**: The remaining data is split 80/10/10 into train/val/test via stratified shuffle split.
9. **Sign filtering**: Each dataset filters events by `signal_weight_sign` / `background_weight_sign` (except in inference mode, where all events are kept).

## Provided Configs

The three provided configs are structurally identical. The main differences are:

| Aspect                     | `config_1D_coupling`        | `config_1D_mass`             | `config_2D`                        |
|----------------------------|-----------------------------|------------------------------|------------------------------------|
| H5 input                   | `Coupling_Inference_negW.h5`| `Mass_Inference_negW.h5`     | `2D_Inference_negW.h5`            |
| Parameter(s)               | `coupling`                  | `hmass`                      | `coupling` × `hmass`              |
| Signal weight sign         | `negative`                  | `positive`                   | `negative`                        |
| Inference truth             | `coupling=1.1`              | `hmass=870`                  | `coupling=0.5, hmass=870`         |

When creating a new config, keep the same section layout and change only the dataset paths, parameter definitions, and mode-specific output/checkpoint locations.
