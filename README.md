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
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=input_plots

python PeakDeepMaster.py --config-name config_1D_mass general.mode=train
python PeakDeepMaster.py --config-name config_2D general.mode=train
python PeakDeepMaster.py --config-name config_2D_width general.mode=train
```

### Available modes

| Mode          | Description |
|---------------|-------------|
| `train`       | Fit a new `LLHRatioEstimator` model and save checkpoints. |
| `predict`     | Run prediction on the test split and export a CSV with scores, labels, and categories. |
| `performance` | Evaluate test and holdout splits, producing ROC curves, score distributions, and summary CSVs. |
| `inference`   | Scan over parameter values using the RoSMM method and compute an $L^2$ statistic to find the best-fit parameter(s). |
| `input_plots` | Generate per-parameter-point distribution plots for all variables listed in `input_plots.variables`. Reads directly from the H5 file (no model needed). |

## Project Structure

```
PeakDeepMaster.py          # Hydra entry point — dispatches to the selected mode.
configs/
  config_1D_coupling.yaml  # 1D inference on coupling strength.
  config_1D_mass.yaml      # 1D inference on heavy Higgs mass.
  config_2D.yaml           # Joint 2D inference on coupling × mass.
  config_2D_width.yaml     # 2D inference with an additional split-only width parameter.
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
    InputPlots.py           # Input-variable distribution plots per parameter point.
    PseudoExperiments.py    # Pseudo-experiment-based parameter uncertainty estimation.
    utils.py                # Logging setup, seed, device selection, checkpoint loading, path resolution.
```

## Configuration Layout

The configs in `configs/` share the same top-level structure:

- `general`: run mode and global runtime settings.
- `logging`: console/file/TensorBoard logging behaviour.
- `dataset`: input file, feature definitions, parameter grids, weights, caching, and dataloader settings.
- `model`: network architecture.
- `train`: optimiser and training controls.
- `predict`: checkpoint and CSV output settings.
- `performance`: checkpoint and plot output settings.
- `inference`: RoSMM inference workflow settings.
- `input_plots`: variable distribution plotting settings.

### `general`

| Key      | Description |
|----------|-------------|
| `mode`   | One of `train`, `predict`, `performance`, `inference`, `input_plots`. |
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
| `cache`                    | Optional dataset caching block (see [Dataset Caching](#dataset-caching)). |
| `remove_from_data_scaling` | Optional list of parameter-point tuples to exclude from scaler fitting (see [Scaler Exclusion](#scaler-exclusion)). |
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
| `split_only`           | If `true`, this parameter is used **only** for data splitting and holdout selection — it is **not** fed to the network as $\theta$. Useful for nuisance parameters like width that define distinct dataset slices but should not be inferred. Default: `false`. |
| `skip_label_in_plots`  | If `true`, omit this parameter from legend labels in plots. Useful for `split_only` parameters. Default: `false`. |
| `add_to_holdout`       | List of full N-dimensional parameter-point tuples to force into the holdout set, even if they would normally be in the training set. Written as a list of lists on the parameter spec that carries it. |
| `remove_from_holdout`  | List of full N-dimensional parameter-point tuples to force out of the holdout set back into training. Inverse of `add_to_holdout`. |

**Multiple parameters are supported.** The `config_2D.yaml` demonstrates joint inference over `coupling` and `hmass`, where the training grid is the Cartesian product of both parameter axes.

**Split-only parameters.** The `config_2D_width.yaml` demonstrates using `hwidth` as a `split_only` parameter: width values partition the data for holdout selection but are never passed to the neural network. At least one parameter must **not** be `split_only`.

Examples across configs:

| Config               | Parameter(s)                  | Training values                                  | Holdout values  |
|----------------------|-------------------------------|--------------------------------------------------|-----------------|
| `config_1D_coupling` | `coupling`                    | `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]`    | `[0.5, 1.1]`   |
| `config_1D_mass`     | `hmass`                       | `[600, 700, 800, 900, 1000]`                     | `[650, 870]`    |
| `config_2D`          | `coupling` × `hmass`          | `[0.2, …, 1.6]` × `[600, …, 1000]`             | `[0.5, 1.1]` × `[650, 870]` |
| `config_2D_width`    | `coupling` × `hmass` × `hwidth` (split-only) | `[0.2, …, 1.6]` × `[600, …, 1000]` × `[2.0]` | `[0.5, 0.7, 1.1]` × `[650, 870, 950]` × `[5.0, 10.0, 15.0, 30.0]` |

##### `add_to_holdout` / `remove_from_holdout`

These options give fine-grained control over which parameter points go into the holdout set. Each entry is the full parameter-point tuple (all parameter dimensions, in order). Place these keys on whichever parameter spec is convenient — they are collected across all specs and merged.

```yaml
parameters:
  - EVENT_GLOBAL: coupling
    values_for_training: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    values_for_testing: [0.5, 1.1]
  - EVENT_GLOBAL: hmass
    values_for_training: [600.0, 700.0, 800.0, 900.0, 1000.0]
    values_for_testing: [650.0, 870.0]
  - EVENT_GLOBAL: hwidth
    values_for_training: [2.0]
    values_for_testing: [5.0, 10.0, 15.0, 30.0]
    split_only: true
    add_to_holdout:
      - [0.7, 950.0, 2.0]   # Force this training-grid point into holdout
```

#### Background augmentation

Background events are augmented across all training parameter points. For each background event, copies are created with every training parameter value, so the background learns to be parameter-independent. Weights are normalised per category and sign before training.

#### Dataset Caching

Processing large H5 files is expensive. The `dataset.cache` block lets you pickle the fully processed datasets (after augmentation, scaling, splitting) and reload them in subsequent runs:

```yaml
dataset:
  cache:
    save_path: /data/pickled/my_dataset.pkl   # Write cache here after processing
    load_path: /data/pickled/my_dataset.pkl   # Load cache from here if it exists
```

| Key         | Description |
|-------------|-------------|
| `save_path` | Path to write the pickle cache after data processing. Created if absent. |
| `load_path` | Path to read a previously saved cache. If the file exists, all data processing is skipped. |

Both keys are optional and independent. A typical workflow is to set `save_path` on the first run, then switch to `load_path` for subsequent runs.

#### Scaler Exclusion

When running inference on parameter points that were **not** present during training, those extra events should be excluded from the scaler fitting so the transformation remains consistent with the originally trained model. Use `remove_from_data_scaling` to list parameter-point tuples that should be excluded:

```yaml
dataset:
  remove_from_data_scaling:
    - [0.5]       # Exclude coupling=0.5 events from scaler fitting
    - [1.1]       # Exclude coupling=1.1 events from scaler fitting
```

Each entry is a list of parameter values (one per parameter dimension, in order). If this key is **not set**, a warning is logged — the scaler will be fitted on all loaded events.

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

The input dimension is `len(observables) + len(model_parameters)` (`x_dim + theta_dim`), where `model_parameters` excludes any `split_only` parameters. The output is a single logit passed through `BCEWithLogitsLoss` (weighted by absolute event weights).

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
| `truth_parameters`   | Mapping of parameter name → true value. Must include **all** parameters (including `split_only` ones) so the correct holdout slice can be identified. E.g., `{coupling: 0.5, hmass: 870.0}` or `{coupling: 0.7, hmass: 950.0, hwidth: 30.0}`. |
| `scan_parameters`    | List of `{name, min, max, n_points}` entries defining each scan axis. Only **model** parameters (non-`split_only`) should be listed. The scan grid is the Cartesian product. |

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

### `input_plots`

Used when `general.mode=input_plots`. Generates distribution plots of raw H5 variables overlaid for each parameter point, without requiring a trained model.

| Key                     | Description |
|-------------------------|-------------|
| `output_dir`            | Directory for output PDFs. |
| `font_size`             | Matplotlib font size for all plots (default: 16). |
| `n_bins`                | Default number of histogram bins (default: 100). |
| `density`               | If `true`, normalise histograms to density (default: `false`). |
| `signal_plus_background`| If `true`, produce additional signal+background overlay plots per parameter point (default: `false`). |
| `log_scale`             | If `true`, produce additional log-scale versions of S+BG plots (default: `false`). |
| `variables`             | List of variable specifications to plot (see below). |

Each entry in `variables` describes one variable to plot:

| Key             | Description |
|-----------------|-------------|
| `group`         | H5 group name (e.g., `EVENT_PARTICLES`, `EVENT_GLOBAL`). |
| `name`          | Variable name inside the group. |
| `display_name`  | Human-readable label used in plot titles. |
| `x_min`, `x_max`| Histogram x-axis range. |
| `x_label`       | X-axis label (supports LaTeX). Also used as the display name for this variable in inference plots. |
| `y_label`       | Y-axis label. |
| `use_weights`   | If `false`, plot unweighted distributions. Default: `true`. When `false`, S+BG plots are skipped for this variable. |
| `n_bins`        | Per-variable override for number of bins. |
| `density`       | Per-variable override for density normalisation. |
| `log_scale`     | Per-variable override for log-scale S+BG plots. |

Example:

```yaml
input_plots:
  output_dir: results/input_plots/1D_coupling/
  font_size: 16
  n_bins: 100
  density: true
  signal_plus_background: true
  log_scale: true
  variables:
    - group: EVENT_GLOBAL
      name: mttbar
      display_name: "Invariant mass $t_1 t_2$ system"
      x_min: 500
      x_max: 1200
      x_label: "$m(t\\bar{t})$ [GeV]"
      y_label: "Density"
    - group: EVENT_GLOBAL
      name: lumiWeight
      display_name: "lumiWeight"
      x_min: -2
      x_max: 2
      use_weights: false
```

---

## Dataset Creation Workflow

This section details the full data pipeline from H5 file to PyTorch tensors, including every transformation and conversion applied to the data. The pipeline is implemented in `PeakDeepMasterDataModule.setup()` and the helpers it calls.

### Step 1 — H5 Loading

Events are read from an HDF5 file with the following structure:

```
INPUTS/
  EVENT_PARTICLES/
    top1_pt, top1_eta, top2_pt, top2_eta, ...
  EVENT_GLOBAL/
    mttbar, coupling, hmass, hwidth, lumiWeight, ...
LABELS/
  CLASS          # 0 = background, 1 = signal
```

The code reads the class labels from `LABELS/CLASS` and reads each configured observable, parameter, and weight from `INPUTS/<GROUP>/<variable>`.

### Step 2 — Unique Parameter Points and Subsampling

The parameter columns are stacked into a matrix, and all unique parameter combinations are identified (rounded to 3 decimal places for numerical stability). If `max_events_per_parameter` is set to a positive integer, a random subsample of at most that many events is drawn per parameter point. This controls memory usage for large datasets.

### Step 3 — Data Structuring

Observables, parameters, and the weight column are concatenated into a single feature matrix $X$ with shape `(n_events, n_observables + n_parameters + 1)`. Column order is:

```
[ observable_1, observable_2, ..., param_1, param_2, ..., weight ]
```

A label matrix $y$ with shape `(n_events, 2)` is built as `[class, category]`, where `category` is an integer ID assigned to each unique parameter point.

### Step 4 — Weight Normalisation (train mode only)

When `general.mode=train`, event weights are normalised **per category and per sign**:

- For each parameter-point category, positive weights are divided by the sum of positive weights in that category, and negative weights are divided by the sum of absolute negative weights.
- Background events (category 0) are normalised by the total weight sum without sign splitting.

This ensures balanced loss contributions across parameter points and weight signs.

### Step 5 — Background Augmentation

Background events are replicated across **every** training parameter point in the Cartesian grid. For each background event, one copy is created per training parameter point, with the parameter columns overwritten to match that point. The original background event's parameter values are replaced with the first training point in the grid; new copies are appended for all remaining points.

This makes the background distribution independent of the parameter conditioning, so the network learns that the background shape does not depend on $\theta$.

After augmentation, the number of background events is multiplied by `len(training_parameter_grid)`.

### Step 6 — Feature Scaling

A scikit-learn `ColumnTransformer` is built from the configured transformations:

| Column type    | Default transformation | Effect |
|----------------|----------------------|--------|
| Observables    | Per-observable config | `LogScaler` → `log(x + 1e-3)`, `StandardScaler` → zero mean / unit variance, `MinMaxScaler` → [0, 1] |
| Parameters     | `MinMaxScaler`       | Scaled to [0, 1] based on the range seen during fitting. |
| Weight         | `passthrough`        | No transformation — weights are left as-is. |

The scaler is **fitted** on the data and then used to **transform** all events (including holdout). If `remove_from_data_scaling` is set, events matching those parameter points are excluded from `scaler.fit()` but are still transformed — this keeps the scaler consistent with a previously trained model when extra holdout points are present in the data file.

All parameters (including `split_only` ones) are scaled. However, only non-`split_only` parameter column indices are recorded as model input columns (`parameter_column_indices`).

### Step 7 — Holdout Separation

Signal events at parameter values listed in `values_for_testing` are separated into the holdout set. The holdout mask is built by checking **each** parameter column independently: any event where **any** parameter value matches a holdout value is flagged. Additionally:

- `add_to_holdout` tuples are OR-ed into the mask (adding specific full-dimensional points).
- `remove_from_holdout` tuples are AND-NOT-ed out of the mask (removing specific points back to training).

Only signal events end up in holdout — if any background events are flagged, an error is raised. Holdout events are stored separately and are never included in the train/val/test splits.

### Step 8 — Train / Validation / Test Splitting

The remaining (non-holdout) data is split using `StratifiedShuffleSplit`:

1. **80% train**, **20% temp** — stratified by the combined `[class, category]` label.
2. The 20% temp is split **50/50** into validation and test — also stratified.

This yields an **80/10/10** split. After splitting, background labels are reset to `[0, 0]` (category 0) since the augmented category assignments are no longer needed.

### Step 9 — Weight-Sign Filtering

Each split (train, val, test) is filtered by `train.signal_weight_sign` and `train.background_weight_sign`:

- If `signal_weight_sign: positive`, only signal events with non-negative weights are kept.
- If `signal_weight_sign: negative`, only signal events with negative weights are kept.
- Same logic for background events using `background_weight_sign`.

This filtering is **skipped** for the holdout set and in `inference` mode (stage `"inference"`), where all events are kept regardless of weight sign.

### Step 10 — Tensor Conversion

The filtered NumPy arrays are converted to `torch.float32` `TensorDataset` objects ready for PyTorch `DataLoader`s.

### Summary Diagram

```
H5 file
  │
  ├─ Read LABELS/CLASS → labels (0/1)
  ├─ Read INPUTS/<GROUP>/<var> for each observable, parameter, weight
  │
  ▼
Subsample (max_events_per_parameter)
  │
  ▼
Structure into X = [obs₁ … obsₙ | param₁ … paramₖ | weight]
               y = [class, category]
  │
  ├─ [train mode] Normalise weights per category and sign
  │
  ▼
Augment background across all training parameter points
  │
  ▼
Fit scaler (optionally excluding remove_from_data_scaling points)
Transform X
  │
  ▼
Separate holdout (values_for_testing ± add/remove_from_holdout)
  │
  ├─ Holdout set → stored as-is (no sign filtering)
  │
  ▼
StratifiedShuffleSplit → 80% train / 10% val / 10% test
  │
  ├─ Reset background categories to 0
  │
  ▼
Filter by signal_weight_sign / background_weight_sign
  │
  ▼
Convert to TensorDataset (torch.float32)
```

---

## Provided Configs

The four provided configs are structurally identical. The main differences are:

| Aspect                     | `config_1D_coupling`        | `config_1D_mass`             | `config_2D`                        | `config_2D_width`                      |
|----------------------------|-----------------------------|------------------------------|------------------------------------|----------------------------------------|
| H5 input                   | `Coupling_Inference_negW.h5`| `Mass_Inference_negW.h5`     | `2D_Inference_negW.h5`            | External width dataset                 |
| Parameter(s)               | `coupling`                  | `hmass`                      | `coupling` × `hmass`              | `coupling` × `hmass` × `hwidth` (split-only) |
| Signal weight sign         | `positive`                  | `negative`                   | `negative`                        | `negative`                             |
| Inference truth             | `coupling=1.1`              | `hmass=650`                  | `coupling=0.5, hmass=650`         | `coupling=0.7, hmass=950, hwidth=30`   |
| Split-only parameters       | —                           | —                            | —                                 | `hwidth`                               |
| `add_to_holdout`            | —                           | —                            | —                                 | `[0.7, 950.0, 2.0]`                   |
| `remove_from_data_scaling`  | `[0.5], [1.1]`              | —                            | —                                 | Several width-related points           |

When creating a new config, keep the same section layout and change only the dataset paths, parameter definitions, and mode-specific output/checkpoint locations.
