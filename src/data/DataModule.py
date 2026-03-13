import h5py
import logging
import lightning as L
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.DataHelpers import (
    augment_data_for_background,
    build_indices_per_parameter_point,
    build_parameter_grid,
    extract_parameter_axes,
    feature_key,
    get_unique_parameter_points,
    holdout_mask_from_parameter_matrix,
    normalize_feature_specs,
    norm_weights_per_category_and_sign,
    parameter_name_from_spec,
    parse_feature_spec,
    structure_data,
)
from src.data.DataScaler import build_feature_scaler
from src.utils.utils import resolve_runtime_path


logger = logging.getLogger(__name__)


class PeakDeepMasterDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_h5_path = resolve_runtime_path(cfg.dataset.input_h5_path)
        self.max_events_per_parameter = cfg.dataset.max_events_per_parameter
        self.train_num_workers = cfg.dataset.train.num_workers
        self.val_num_workers = cfg.dataset.val.num_workers
        self.train_batch_size = cfg.dataset.train.batch_size
        self.val_batch_size = cfg.dataset.val.batch_size

        self.observables_config = normalize_feature_specs(cfg.dataset.observables)
        logger.debug("Observables configuration: %s", self.observables_config)
        parameter_specs = normalize_feature_specs(cfg.dataset.parameters)
        weight_specs = normalize_feature_specs(cfg.dataset.weights)

        if len(parameter_specs) == 0:
            raise ValueError("At least one entry must be provided in dataset.parameters")
        if len(weight_specs) != 1:
            raise ValueError("Exactly one entry is currently supported in dataset.weights")

        self.parameter_specs = parameter_specs
        self.parameter_spec = parameter_specs[0]
        logger.debug("Parameter specs: %s", self.parameter_specs)
        self.weight_spec = weight_specs[0]
        self.parameter_names = [parameter_name_from_spec(spec) for spec in self.parameter_specs]
        self.parameter_axes = {
            name: axis
            for name, axis in zip(self.parameter_names, extract_parameter_axes(self.parameter_specs, "values_for_training"))
        }
        self.holdout_parameter_axes = {
            name: axis
            for name, axis in zip(self.parameter_names, extract_parameter_axes(self.parameter_specs, "values_for_testing"))
        }
        self.training_parameter_grid = build_parameter_grid(self.parameter_specs, "values_for_training")
        if len(self.training_parameter_grid) == 0:
            raise ValueError("values_for_training must be provided in dataset.parameters")
        logger.debug("Parameter axes: %s", self.parameter_axes)
        logger.debug("Holdout parameter axes: %s", self.holdout_parameter_axes)
        logger.debug("Training parameter grid: %s", self.training_parameter_grid)


        self.random_seed = int(getattr(cfg.dataset, "random_seed", cfg.general.seed))

        self.signal_weight_sign = cfg.train.signal_weight_sign
        self.background_weight_sign = cfg.train.background_weight_sign

        self.scaler = None
        self.parameters_category_dict = {}
        self.parameter_point_to_category = {}
        self.category_to_parameter_point = {}
        self.feature_index_map = {}
        self.transformed_feature_keys = []
        self.parameter_column_indices = []
        self.parameter_transformer_names = []

        self.X_holdout = None
        self.y_holdout = None
        self.holdout_dataset = None
        self._has_setup = False

    def _build_stratify_labels(self, y: np.ndarray) -> np.ndarray:
        labels = y[:, 0].astype(int)
        categories = y[:, 1].astype(int)
        return np.asarray([f"{label}_{category}" for label, category in zip(labels, categories)])

    def setup(self, stage: str | None = None):
        if self._has_setup:
            return
        self._has_setup = True

        logger.info("Preparing data for stage = %s.\nWith data from file = %s", self.cfg.general.mode, self.input_h5_path)

        with h5py.File(self.input_h5_path, "r") as data_file:
            labels_dataset = data_file["LABELS"]["CLASS"][:]
            parameter_arrays = []
            for parameter_spec in self.parameter_specs:
                parameter_group, parameter_variable = parse_feature_spec(parameter_spec)
                parameter_arrays.append(data_file["INPUTS"][parameter_group][parameter_variable][:])

            parameter_matrix = np.column_stack(parameter_arrays)
            parameter_points = get_unique_parameter_points(parameter_matrix)
            logger.debug("Unique parameter points in dataset: %s", parameter_points)
            missing_training_points = sorted(set(self.training_parameter_grid) - set(parameter_points))
            if missing_training_points:
                logger.warning(
                    "Configured training grid contains %d parameter point(s) missing from the dataset: %s",
                    len(missing_training_points),
                    missing_training_points,
                )
            indices_per_parameter = build_indices_per_parameter_point(
                parameter_matrix=parameter_matrix,
                parameter_points=parameter_points,
                max_events_per_parameter=self.max_events_per_parameter,
                random_seed=self.random_seed,
            )
            training_indices = np.concatenate(list(indices_per_parameter.values()))
            logger.debug("Number of parameter indices: %d", len(indices_per_parameter))
            
            logger.debug("Structuring the data..")
            X, y, self.parameter_point_to_category, self.feature_index_map = structure_data(
                data_file=data_file,
                labels_dataset=labels_dataset,
                parameter_points=parameter_points,
                observables_config=self.observables_config,
                parameter_specs=self.parameter_specs,
                weight_spec=self.weight_spec,
                training_indices=training_indices,
            )
            self.parameters_category_dict = dict(self.parameter_point_to_category)
            self.category_to_parameter_point = {
                category: point for point, category in self.parameter_point_to_category.items()
            }
            logger.debug("Data[0] example: %s", X[0])
            logger.debug("Data[1000000] example: %s", X[1000000])

        weight_group, weight_variable = parse_feature_spec(self.weight_spec)
        parameter_feature_keys = []
        parameter_column_indices = []
        for parameter_spec in self.parameter_specs:
            param_group, param_variable = parse_feature_spec(parameter_spec)
            key = feature_key(param_group, param_variable)
            parameter_feature_keys.append(key)
            parameter_column_indices.append(self.feature_index_map[key])
        weight_column_index = self.feature_index_map[feature_key(weight_group, weight_variable)]

        logger.debug("Normalizing weights...")
        X[:, weight_column_index] = norm_weights_per_category_and_sign(X[:, weight_column_index], y[:, -1])
        logger.debug("Data[0] example: %s", X[0])
        logger.debug("Data[1000000] example: %s", X[1000000])

        logger.debug("Augmenting data for background...")
        X, y = augment_data_for_background(
            X,
            y,
            self.training_parameter_grid,
            parameter_column_indices=parameter_column_indices,
        )
        logger.debug("Data[0] example: %s", X[0])
        logger.debug("Data[1000000] example: %s", X[1000000])
        logger.debug("Data[5000000] example: %s", X[5000000])

        parameter_matrix_for_holdout = X[:, parameter_column_indices].copy()

        logger.debug("Building feature scaler...")
        self.scaler, self.transformed_feature_keys = build_feature_scaler(
            self.observables_config,
            self.parameter_specs,
            self.weight_spec,
        )

        logger.debug("Fitting and transforming features...")
        X = self.scaler.fit_transform(X)
        logger.debug("Data[0] example: %s", X[0])
        logger.debug("Data[1000000] example: %s", X[1000000])
        logger.debug("Data[5000000] example: %s", X[5000000])

        self.parameter_column_indices = [self.transformed_feature_keys.index(key) for key in parameter_feature_keys]
        self.parameter_column_index = self.parameter_column_indices[0]
        self.parameter_transformer_names = [f"param_{name}" for name in self.parameter_names]
        self.weight_column_index = self.transformed_feature_keys.index(feature_key(weight_group, weight_variable))

        self.observable_column_indices = []
        for obs in self.observables_config:
            obs_group, obs_name = parse_feature_spec(obs)
            key = feature_key(obs_group, obs_name)
            self.observable_column_indices.append(self.transformed_feature_keys.index(key))

        self.x_column_indices = list(self.observable_column_indices)
        logger.debug("Observable column indices: %s", self.observable_column_indices)
        logger.debug("Parameter column indices: %s", self.parameter_column_indices)
        logger.debug("Weight column index: %d", self.weight_column_index)

        logger.debug("Removing the holdout datasets with parameter axes %s", self.holdout_parameter_axes)
        holdout_mask = holdout_mask_from_parameter_matrix(parameter_matrix_for_holdout, self.parameter_specs)
        logger.debug("Holdout mask length: %d, holdout count: %d", len(holdout_mask), np.sum(holdout_mask))

        self.X_holdout = X[holdout_mask]
        self.y_holdout = y[holdout_mask]

        X_model = X[~holdout_mask]
        y_model = y[~holdout_mask]
        logger.debug("Data_wo_holdout[0] example: %s", X_model[0])
        logger.debug("Data_wo_holdout[1000000] example: %s", X_model[1000000])
        logger.debug("Data_wo_holdout[5000000] example: %s", X_model[5000000])

        logger.info("Splitting data into train, val, and test sets...")
        stratify_model = y_model[:, 1] # self._build_stratify_labels(y_model)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.cfg.general.seed)
        train_idx, temp_idx = next(sss.split(X_model, stratify_model))

        X_train, y_train = X_model[train_idx], y_model[train_idx]
        X_temp, y_temp = X_model[temp_idx], y_model[temp_idx]
        stratify_temp = self._build_stratify_labels(y_temp)

        logger.debug("X_train[0] example: %s", X_train[0])
        logger.debug("X_train[1000000] example: %s", X_train[1000000])
        logger.debug("X_train[5000000] example: %s", X_train[5000000])

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            shuffle=True,
            stratify=stratify_temp,
            random_state=self.cfg.general.seed,
        )

        logger.debug("Exporting to TensorDatasets...")
        self.train_dataset = self._to_dataset(X_train, y_train, stage=stage)
        self.val_dataset = self._to_dataset(X_val, y_val, stage=stage)
        self.test_dataset = self._to_dataset(X_test, y_test, stage=stage)
        self.holdout_dataset = self._to_dataset(self.X_holdout, self.y_holdout, stage=stage)

        logger.info(
            "Prepared datasets: train = %d, val = %d, test = %d, holdout = %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
            0 if self.X_holdout is None else len(self.X_holdout),
        )

    def _to_dataset(self, X: np.ndarray, y: np.ndarray, stage: str) -> TensorDataset:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if stage == "inference":
            return TensorDataset(X_tensor, y_tensor)

        filtered_mask = self._sign_mask(X_tensor, y_tensor)
        X_tensor = X_tensor[filtered_mask]
        y_tensor = y_tensor[filtered_mask]
        return TensorDataset(X_tensor, y_tensor)

    def _sign_mask(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        signal_mask = y[:, 0] == 1
        background_mask = y[:, 0] == 0
        weight_values = X[:, self.weight_column_index]

        if self.signal_weight_sign == "positive":
            signal_weight_mask = weight_values >= 0
        else:
            signal_weight_mask = weight_values < 0

        if self.background_weight_sign == "positive":
            background_weight_mask = weight_values >= 0
        else:
            background_weight_mask = weight_values < 0

        return (signal_mask & signal_weight_mask) | (background_mask & background_weight_mask)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            persistent_workers=self.train_num_workers > 0,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            persistent_workers=self.val_num_workers > 0,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_eval_dataset(self, split: str) -> TensorDataset:
        if split == "test":
            return self.test_dataset
        if split == "holdout":
            return self.holdout_dataset
        raise ValueError(f"Unsupported evaluation split: {split}")
