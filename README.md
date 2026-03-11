# PeakDeepMaster

PeakDeepMaster mirrors the `VBFTransformer` project layout and applies PyTorch Lightning to the coupling-estimation workflow with negative event weights from `Coupling_Inference_negW.h5`.

## Setup

```bash
conda env create -f requirements.yaml
conda activate peak_deep_master
pip install -e .
```

## Modes

The runner script uses Hydra:

```bash
python PeakDeepMaster.py --config-name config general.mode=train
python PeakDeepMaster.py --config-name config general.mode=predict
python PeakDeepMaster.py --config-name config general.mode=performance
python PeakDeepMaster.py --config-name config general.mode=infer
```

## Problem Mapping

- Reads one H5 source with signal/background + coupling labels.
- Reproduces notebook preprocessing: category mapping, optional background coupling augmentation, signed-weight normalization, coupling conditioning.
- Trains two ratio estimators (positive-weight signal and negative-weight signal) with weighted BCE.
- Provides RoSMM-based shape inference helper for coupling scans.

## Configuration Layout

The framework reads dataset inputs from nested config sections:

- `logging`: global Python logging configuration for console/file output.
- `dataset.input_h5_path`: H5 input file path. Relative paths are resolved from the project root even when Hydra changes the run directory.
- `dataset.observables`: list of features used as `x` (each with `transformation`).
- `dataset.parameters[0]`: coupling feature (`values_for_training` and `values_for_testing`).
- `dataset.weights[0]`: signed event weights.
- `dataset.train` and `dataset.val`: dataloader worker and batch-size settings.

Notes:

- `EVENT_PATICLES` (typo) is accepted and internally normalized to `EVENT_PARTICLES`.
- Column positions and the model input size are inferred from configured feature order.
- Logging uses the standard library `logging` module and can be redirected to a file through `logging.file`.
- Console logs support colored level headers through `logging.colors` (`auto`, `true`, or `false`), while file logs remain plain text.
