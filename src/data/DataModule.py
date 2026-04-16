import h5py
import logging
import pickle
from pathlib import Path

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
    is_split_only,
    normalize_feature_specs,
    normalize_parameter_point,
    norm_weights_per_category_and_sign,
    parameter_name_from_spec,
    parse_feature_spec,
    parse_parameter_point_tuple_to_list,
    structure_data,
    parse_parameter_point_tuple_to_list
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

        # Model parameters: those actually fed as theta to the network (excludes split_only).
        self.model_parameter_specs = [spec for spec in self.parameter_specs if not is_split_only(spec)]
        self.model_parameter_names = [parameter_name_from_spec(spec) for spec in self.model_parameter_specs]
        if len(self.model_parameter_specs) == 0:
            raise ValueError("At least one parameter must not be split_only")
        logger.debug("Model parameter names: %s", self.model_parameter_names)

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

        # Parse add_to_holdout / remove_from_holdout tuples from parameter specs.
        self.add_to_holdout = self._parse_holdout_tuples("add_to_holdout")
        self.remove_from_holdout = self._parse_holdout_tuples("remove_from_holdout")
        if self.add_to_holdout:
            logger.info("Explicitly adding to holdout: %s", self.add_to_holdout)
        if self.remove_from_holdout:
            logger.info("Explicitly removing from holdout: %s", self.remove_from_holdout)


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

    def _parse_holdout_tuples(self, key: str) -> list[tuple[float, ...]] | None:
        """Collect add_to_holdout / remove_from_holdout lists from parameter specs."""
        raw_lists = [spec.get(key, None) for spec in self.parameter_specs]
        if all(v is None for v in raw_lists):
            return None
        # Each spec may carry a list of partial values.  We need the full
        # N-dimensional parameter points, so the user supplies tuples directly
        # on the *first* parameter that contains the key.  Collect from all
        # specs and merge – but the convention is to write the full tuples in
        # the first spec that has the key.
        tuples: list[tuple[float, ...]] = []
        for raw in raw_lists:
            if raw is None:
                continue
            for entry in raw:
                point = normalize_parameter_point(list(entry))
                tuples.append(point)
        return tuples if tuples else None

    # -- Dataset cache helpers ------------------------------------------------

    _CACHE_ATTRIBUTES = (
        "scaler",
        "parameters_category_dict",
        "parameter_point_to_category",
        "category_to_parameter_point",
        "feature_index_map",
        "transformed_feature_keys",
        "parameter_column_indices",
        "parameter_transformer_names",
        "weight_column_index",
        "observable_column_indices",
        "x_column_indices",
        "X_holdout",
        "y_holdout",
        "train_dataset",
        "val_dataset",
        "test_dataset",
        "holdout_dataset",
    )

    def _save_cache(self, path: str) -> None:
        """Pickle the processed datasets and metadata to *path*."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        state = {attr: getattr(self, attr) for attr in self._CACHE_ATTRIBUTES}
        with open(cache_path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved dataset cache to %s", cache_path)

    def _load_cache(self, path: str) -> None:
        """Restore processed datasets and metadata from a pickle at *path*."""
        cache_path = Path(path)
        with open(cache_path, "rb") as fh:
            state = pickle.load(fh)
        for attr in self._CACHE_ATTRIBUTES:
            setattr(self, attr, state[attr])
        logger.info("Loaded dataset cache from %s", cache_path)

    # ------------------------------------------------------------------------

    def setup(self, stage: str | None = None):
        if self._has_setup:
            return
        self._has_setup = True

        # Check for dataset cache configuration.
        cache_cfg = getattr(self.cfg.dataset, "cache", None)
        load_path = None
        save_path = None
        if cache_cfg is not None:
            load_path = getattr(cache_cfg, "load_path", None)
            save_path = getattr(cache_cfg, "save_path", None)
            if load_path is not None:
                load_path = resolve_runtime_path(str(load_path))
            if save_path is not None:
                save_path = resolve_runtime_path(str(save_path))

        if load_path is not None and Path(load_path).is_file():
            logger.info("Loading cached dataset from %s", load_path)
            self._load_cache(load_path)
            return

        logger.info("Preparing data for stage = %s.\nWith data from file = %s", self.cfg.general.mode, self.input_h5_path)

        with h5py.File(self.input_h5_path, "r") as data_file:
            logger.info("Reading class labels column.")
            labels_dataset = data_file["LABELS"]["CLASS"][:]
            parameter_arrays = []
            logger.info("Reading parameter columns.")
            for parameter_spec in self.parameter_specs:
                parameter_group, parameter_variable = parse_feature_spec(parameter_spec)
                parameter_arrays.append(data_file["INPUTS"][parameter_group][parameter_variable][:])

            parameter_matrix = np.column_stack(parameter_arrays)
            logger.info("Finding unique parameter combinations")
            parameter_points = get_unique_parameter_points(parameter_matrix)
            logger.debug("Unique parameter points in dataset: %s", parameter_points)
            missing_training_points = sorted(set(self.training_parameter_grid) - set(parameter_points))
            if missing_training_points:
                logger.warning(
                    "Configured training grid contains %d parameter point(s) missing from the dataset: %s",
                    len(missing_training_points),
                    missing_training_points,
                )
            logger.info("Building row indices for each parameter point.")
            indices_per_parameter = build_indices_per_parameter_point(
                parameter_matrix=parameter_matrix,
                parameter_points=parameter_points,
                max_events_per_parameter=self.max_events_per_parameter,
                random_seed=self.random_seed,
            )
            logger.info("Concatenating training indices across parameter points.")
            filter_indices = np.concatenate(list(indices_per_parameter.values()))
            logger.debug("Number of parameter indices: %d", len(indices_per_parameter))
            
            logger.info("Structuring the data..")
            X, y, self.parameter_point_to_category, self.feature_index_map = structure_data(
                data_file=data_file,
                labels_dataset=labels_dataset,
                parameter_points=parameter_points,
                observables_config=self.observables_config,
                parameter_specs=self.parameter_specs,
                weight_spec=self.weight_spec,
                filter_indices=filter_indices,
            )
            self.parameters_category_dict = dict(self.parameter_point_to_category)
            self.category_to_parameter_point = {
                category: point for point, category in self.parameter_point_to_category.items()
            }
            for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X):
                    logger.debug("Data[%d] example: %s", event_idx, X[event_idx])
            # Log the number of signal and background events after structuring, and per-parameter as well.
            logger.info("Total structured events: %d", len(X))
            logger.info("  Signal events: %d", np.sum(y[:, 0] == 1))
            logger.info("  Background events: %d", np.sum(y[:, 0] == 0))
            for point, category in self.parameter_point_to_category.items():
                point_mask = y[:, 1] == category
                n_point_events = np.sum(point_mask)
                n_point_signal = np.sum((y[:, 0] == 1) & point_mask)
                n_point_background = np.sum((y[:, 0] == 0) & point_mask)
                logger.info(
                    "Parameter point %s (category %d): total=%d, signal=%d, background=%d",
                    parse_parameter_point_tuple_to_list(point),
                    category,
                    n_point_events,
                    n_point_signal,
                    n_point_background,
                )

        weight_group, weight_variable = parse_feature_spec(self.weight_spec)
        # All-parameter keys/indices (used for augmentation and holdout).
        all_parameter_feature_keys = []
        all_parameter_column_indices = []
        for parameter_spec in self.parameter_specs:
            param_group, param_variable = parse_feature_spec(parameter_spec)
            key = feature_key(param_group, param_variable)
            all_parameter_feature_keys.append(key)
            all_parameter_column_indices.append(self.feature_index_map[key])

        # Model-parameter keys (only those fed as theta to the network).
        model_parameter_feature_keys = []
        for parameter_spec in self.model_parameter_specs:
            param_group, param_variable = parse_feature_spec(parameter_spec)
            model_parameter_feature_keys.append(feature_key(param_group, param_variable))

        weight_column_index = self.feature_index_map[feature_key(weight_group, weight_variable)]

        if self.cfg.general.mode == "train":
            logger.info("Normalizing weights...")
            X[:, weight_column_index] = norm_weights_per_category_and_sign(X[:, weight_column_index], y[:, -1])
        for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X):
                    logger.debug("Data[%d] example: %s", event_idx, X[event_idx])

        logger.info("Augmenting data for background... this will multiply the number of BG events by %d", len(self.training_parameter_grid))
        X, y = augment_data_for_background(
            X,
            y,
            self.training_parameter_grid,
            parameter_column_indices=all_parameter_column_indices,
            parameter_point_to_category=self.parameter_point_to_category,
        )
        for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X):
                    logger.debug("Data[%d] example: %s", event_idx, X[event_idx])

        parameter_matrix_for_holdout = X[:, all_parameter_column_indices].copy()

        # Scale the features
        logger.info("Building feature scaler...")
        self.scaler, self.transformed_feature_keys = build_feature_scaler(
            self.observables_config,
            self.parameter_specs,
            self.weight_spec,
        )
        logger.info("Fitting and transforming features...")
        # By default fit the scaler on all events; optionally exclude specific parameter points.
        mask_for_transforming = np.ones(len(X), dtype=bool)
        remove_from_scaling_raw = getattr(self.cfg.dataset, "remove_from_data_scaling", None)
        if remove_from_scaling_raw is not None:
            param_values = np.round(X[:, all_parameter_column_indices].copy(), 3)
            remove_from_scaling = [normalize_parameter_point(list(entry)) for entry in remove_from_scaling_raw]
            for point in remove_from_scaling:
                point_match = np.all(np.isclose(param_values, point, atol=1e-3), axis=1)
                mask_for_transforming &= ~point_match
            n_removed = len(X) - int(mask_for_transforming.sum())
            logger.info(
                "Using removal for data scaling."
                "excluded %d events matching %d parameter point(s) from scaler fitting.",
                n_removed, len(remove_from_scaling),
            )
        else:
            logger.warning(
                "dataset.remove_from_data_scaling is not set. Assuming all data fed into this run "
                "was used for data transformation when the model was trained. This might not be true — please check."
            )

        logger.debug("Total events: %d, events used for scaler fitting: %d", len(X), mask_for_transforming.sum())
        self.scaler.fit(X[mask_for_transforming])
        X = self.scaler.transform(X)
        for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X):
                    logger.debug("Data[%d] example: %s", event_idx, X[event_idx])

        self.parameter_column_indices = [self.transformed_feature_keys.index(key) for key in model_parameter_feature_keys]
        self.parameter_transformer_names = [f"param_{name}" for name in self.model_parameter_names]
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
        logger.debug("Holdout parameter matrix shape: %s, and values: %s", parameter_matrix_for_holdout.shape, parameter_matrix_for_holdout)
        holdout_mask = holdout_mask_from_parameter_matrix(
            parameter_matrix_for_holdout,
            self.parameter_specs,
            add_to_holdout=self.add_to_holdout,
            remove_from_holdout=self.remove_from_holdout,
        )
        logger.debug("Holdout mask length: %d, holdout count: %d", len(holdout_mask), np.sum(holdout_mask))

        self.X_holdout = X[holdout_mask]
        self.y_holdout = y[holdout_mask]

        # Check if any of the holdouts is BG. If it is raise an error, because we don't want to hold out any BG points.
        if np.any(self.y_holdout[:, 0] == 0):
            raise ValueError("Holdout set contains background events, which is not allowed. Please check your holdout configuration.")

        X_model = X[~holdout_mask]
        y_model = y[~holdout_mask]
        for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X_model):
                    logger.debug("Data_wo_holdout[%d] example: %s", event_idx, X_model[event_idx])

        logger.info("Splitting data into train, val, and test sets...")
        stratify_model = y_model
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.cfg.general.seed)
        train_idx, temp_idx = next(sss.split(X_model, stratify_model))

        X_train, y_train = X_model[train_idx], y_model[train_idx]
        X_temp, y_temp = X_model[temp_idx], y_model[temp_idx]
        stratify_temp = y_temp

        for event_idx in self.cfg.logging.events_to_log_in_debug:
                if event_idx < len(X):
                    logger.debug("Data[%d] example: %s", event_idx, X_train[event_idx])

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            shuffle=True,
            stratify=stratify_temp,
            random_state=self.cfg.general.seed,
        )

        # Convert background [label,categories] back to [0,0]
        logger.info("Converting background labels back to [0 - class, 0 - category] format...")
        y_train_bg_mask = y_train[:, 0] == 0
        y_train[y_train_bg_mask, 1] = 0
        y_test_bg_mask = y_test[:, 0] == 0
        y_test[y_test_bg_mask, 1] = 0
        y_val_bg_mask = y_val[:, 0] == 0
        y_val[y_val_bg_mask, 1] = 0

        logger.info("Split almost complete, only sign(W) filter remains (if applicable).")
        logger.info("Before filter... train = %d, val = %d, test = %d, holdout = %d",
            X_train.shape[0],
            X_val.shape[0],
            X_test.shape[0],
            0 if self.X_holdout is None else self.X_holdout.shape[0],
        )

        logger.debug("Exporting to TensorDatasets...")
        self.train_dataset = self._to_dataset(X_train, y_train, stage=stage)
        self.val_dataset = self._to_dataset(X_val, y_val, stage=stage)
        self.test_dataset = self._to_dataset(X_test, y_test, stage=stage)
        self.holdout_dataset = self._to_dataset(self.X_holdout, self.y_holdout, stage=stage)

        logger.info(
            "After filter... train = %d, val = %d, test = %d, holdout = %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
            0 if self.X_holdout is None else len(self.X_holdout),
        )

        # Generate report of train, val, test, and holdout counts per parameter point.
        logger.info("Dataset counts per parameter point (after all processing):")
        for point, category in self.parameter_point_to_category.items():
            point_mask_train = y_train[:, 1] == category
            point_mask_val = y_val[:, 1] == category
            point_mask_test = y_test[:, 1] == category
            point_mask_holdout = self.y_holdout[:, 1] == category if self.X_holdout is not None else np.array([False] * len(self.y_holdout))
            n_train = np.sum(point_mask_train)
            n_val = np.sum(point_mask_val)
            n_test = np.sum(point_mask_test)
            n_holdout = np.sum(point_mask_holdout)

            logger.info(
                "Parameter point %s (category %d): train=%d, val=%d, test=%d, holdout=%d",
                parse_parameter_point_tuple_to_list(point),
                category,
                n_train,
                n_val,
                n_test,
                n_holdout,
            )

            # Validate if things make sense.
            # Holdout points should have zero events in train/val/test, and nonzero in holdout. Non-holdout points should have zero events in holdout.
            if n_holdout > 0 and (n_train > 0 or n_val > 0 or n_test > 0):
                logger.warning(
                    "Parameter point %s has holdout events but also has train/val/test events. This may indicate a problem with the holdout configuration.",
                    parse_parameter_point_tuple_to_list(point),
                )
            if n_holdout == 0 and (n_train == 0 and n_val == 0 and n_test == 0):
                logger.warning(
                    "Parameter point %s has no events in holdout but also has no events in train/val/test. This may indicate a problem with the holdout configuration.",
                    parse_parameter_point_tuple_to_list(point),
                )
            # Points used for training should not have holdout events.
            if n_train > 0 and n_holdout > 0:
                logger.warning(
                    "Parameter point %s has events in both train and holdout sets. This may indicate a problem with the holdout configuration.",
                    parse_parameter_point_tuple_to_list(point),
                )
                # Check that the number of validation and test events corresponds to what we expect between a 5% margin.
                expected_val = n_train * 0.1
                expected_test = n_train * 0.1
                if not (expected_val * 0.95 <= n_val <= expected_val * 1.05):
                    logger.warning(
                        "Parameter point %s has %d validation events, which is outside the expected range of [%d, %d] based on a 10%% split of the training events. This may indicate a problem with the data splitting.",
                        parse_parameter_point_tuple_to_list(point),
                        n_val,
                        int(expected_val * 0.95),
                        int(expected_val * 1.05),
                    )
                if not (expected_test * 0.95 <= n_test <= expected_test * 1.05):
                    logger.warning(
                        "Parameter point %s has %d test events, which is outside the expected range of [%d, %d] based on a 10%% split of the training events. This may indicate a problem with the data splitting.",
                        parse_parameter_point_tuple_to_list(point),
                        n_test,
                        int(expected_test * 0.95),
                        int(expected_test * 1.05),
                    )


        if save_path is not None:
            self._save_cache(save_path)

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
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            persistent_workers=self.val_num_workers > 0,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=8,drop_last=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_eval_dataset(self, split: str) -> TensorDataset:
        if split == "test":
            return self.test_dataset
        if split == "holdout":
            return self.holdout_dataset
        raise ValueError(f"Unsupported evaluation split: {split}")
