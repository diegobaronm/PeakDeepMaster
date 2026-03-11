# PeakDeepMaster

This repo performs SBI for heavy Higgs events with strong interference with the SM.

## Setup

```bash
conda env create -f requirements.yaml
conda activate peak_deep_master
pip install -e .
```

## Modes

The runner script uses Hydra:

```bash
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=train
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=predict
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=performance
python PeakDeepMaster.py --config-name config_1D_coupling general.mode=infer

python PeakDeepMaster.py --config-name config_1D_mass general.mode=train
```

## Configuration Layout

The example configs in `configs/config_1D_coupling.yaml` and `configs/config_1D_mass.yaml` follow the same top-level structure:

- `general`: run mode and global runtime settings.
- `logging`: console/file logging behavior.
- `dataset`: input file, feature definitions, parameter scan choices, weights, and dataloader settings.
- `model`: network size and dropout.
- `train`: optimizer and training controls.
- `predict`: checkpoint and CSV output settings.
- `performance`: checkpoint and plot output settings.
- `inference`: settings for the `general.mode=infer` workflow.

### `general`

`general.mode` selects which utility is executed:

- `train`: fit a new model.
- `predict`: run prediction on the test split and export a CSV.
- `performance`: run test metrics and write score/ROC plots.
- `infer`: scan over parameter values and compute the inference statistic.

`general.device` is passed through to Lightning (`cuda`, `cpu`, ...), and `general.seed` is used for reproducible splitting and sampling.

### `logging`

This section is standard Python logging configuration used by `setup_logging`:

- `level`, `format`, `datefmt`: normal logging controls.
- `colors`: colored console output (`auto`, `true`, `false`).
- `file`: optional log file path; `null` keeps logging on the console only.
- `capture_warnings`: redirects Python warnings into the logger.

### `dataset`

This is the most important section because it defines both the H5 input and how columns are mapped into the model.

- `input_h5_path`: H5 file to read. Relative paths are resolved from the project root, so they still work when Hydra changes the run directory.
- `max_events_per_parameter`: cap used when subsampling events for each parameter value.
- `observables`: ordered list of input features used as `x` by the classifier.
- `parameters`: currently exactly one parameter entry is supported.
- `weights`: currently exactly one event-weight entry is supported.
- `train` / `val`: dataloader worker count and batch size for the corresponding loaders.

Feature entries are written as YAML mappings like:

```yaml
- EVENT_PARTICLES: top1_pt
	transformation: LogScaler
```

The non-`transformation` key identifies the H5 location as `GROUP: variable`. In the example above, the code reads `INPUTS/EVENT_PARTICLES/top1_pt`. The supported transformations are:

- `LogScaler`
- `StandardScaler`
- `MinMaxScaler`

The order of `dataset.observables` matters. That order is used to build the model input vector, and `len(dataset.observables)` becomes the network `x` dimension.

`dataset.parameters[0]` defines the SBI parameter being learned. In the attached examples:

- `config_1D_coupling.yaml` uses `EVENT_GLOBAL: coupling`.
- `config_1D_mass.yaml` uses `EVENT_GLOBAL: hmass`.

The parameter entry also controls the split logic:

- `values_for_training`: parameter values kept in the train/val/test splits.
- `values_for_testing`: holdout parameter values removed from training and kept for evaluation/inference.

That means the coupling config trains on `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]` and holds out `[0.5, 1.1]`, while the mass config trains on `[600, 700, 800, 900, 1000]` and holds out `[650, 870]`.

`dataset.weights[0]` supplies the event weights. In both examples this is `EVENT_GLOBAL: lumiWeight`. The datamodule currently requires exactly one parameter feature and one weight feature.

### `model`

This section configures the `RatioEstimatorNet`:

- `hidden_dim`: width of the first hidden layers.
- `dropout`: dropout probability used in the MLP.

### `train`

These values are consumed only when `general.mode=train`:

- `n_epochs`: maximum number of epochs.
- `learning_rate`, `weight_decay`: AdamW optimizer settings.
- `lr_patience`, `lr_factor`: `ReduceLROnPlateau` scheduler settings.
- `signal_weight_sign`, `background_weight_sign`: filter which signed-weight events are kept for each class.
- `compile`: optionally enables `torch.compile`.

### `predict`

Used when `general.mode=predict`:

- `checkpoint_path`: directory containing Lightning checkpoints. The latest checkpoint is selected automatically.
- `output_file`: CSV file written with prediction, label, and category columns.

### `performance`

Used when `general.mode=performance`:

- `checkpoint_path`: checkpoint directory to evaluate.
- `output_dir`: destination for ROC curves, score histograms, and summary CSV files.
- `batch_size`: batch size used for evaluation/prediction when producing plots.

This mode evaluates both the regular test split and the holdout split defined by `dataset.parameters[0].values_for_testing`.

### `inference`

Used when `general.mode=infer`:

- `model_pp_checkpoint` and `model_pn_checkpoint`: checkpoint directories for the positive-positive and positive-negative models.
- `theta_min`, `theta_max`, `n_points`: parameter scan range.
- `output_dir`: reserved result directory for inference outputs.
- `observable`: a label for the observable of interest in the config; the current implementation specifically requires `mttbar` to be present in `dataset.observables`.

For inference, the code builds a reference background sample from the test split, compares it to the holdout signal sample, and scans the configured parameter range to find the best-fit value.

## Config Differences In Practice

The two provided configs are structurally identical. The main differences are:

- the H5 input file,
- the parameter feature (`coupling` vs `hmass`),
- the train/holdout parameter grids,
- the checkpoint/output destinations,
- and the scan range in the `inference` section.

If you create a new config, keep the same section layout and change only the dataset paths, parameter definition, and mode-specific output/checkpoint locations.
