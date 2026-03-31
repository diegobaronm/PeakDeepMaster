import itertools
import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)

PARAMETER_DECIMALS = 3

def normalize_parameter_value(value: float) -> float:
    return round(float(value), PARAMETER_DECIMALS)

def normalize_parameter_point(values) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return tuple(normalize_parameter_value(value) for value in array)

def parse_feature_spec(spec: dict) -> tuple[str, str]:
    ignored_keys = {"transformation", "values_for_training", "values_for_testing", "split_only",
                    "add_to_holdout", "remove_from_holdout"}
    for key, value in spec.items():
        if key in ignored_keys:
            continue
        return str(key), str(value)
    raise ValueError(f"Invalid feature spec: {spec}")


def is_split_only(spec: dict) -> bool:
    """Return True if the parameter spec is marked as split-only."""
    return bool(spec.get("split_only", False))

def normalize_feature_specs(specs) -> list[dict]:
    return [dict(spec) for spec in specs]

def feature_key(group_name: str, variable_name: str) -> str:
    return f"{group_name}:{variable_name}"

def parameter_name_from_spec(spec: dict) -> str:
    _, variable_name = parse_feature_spec(spec)
    return variable_name

def get_feature_from_h5(data_file: h5py.File, group_name: str, variable_name: str) -> np.ndarray:
    return data_file["INPUTS"][group_name][variable_name][:]


def extract_parameter_axes(parameter_specs: list[dict], values_key: str) -> list[list[float]]:
    axes = []
    for spec in parameter_specs:
        axis = [normalize_parameter_value(value) for value in spec.get(values_key, [])]
        if values_key == "values_for_training" and len(axis) == 0:
            name = parameter_name_from_spec(spec)
            raise ValueError(f"values_for_training must be provided for parameter '{name}'")
        axes.append(axis)
    return axes


def build_parameter_grid(parameter_specs: list[dict], values_key: str = "values_for_training") -> list[tuple[float, ...]]:
    axes = extract_parameter_axes(parameter_specs, values_key)
    if len(axes) == 0:
        return []
    return [normalize_parameter_point(point) for point in itertools.product(*axes)]


def get_unique_parameter_points(parameter_matrix: np.ndarray) -> list[tuple[float, ...]]:
    normalized = [normalize_parameter_point(row) for row in np.asarray(parameter_matrix, dtype=float)]
    return sorted(set(normalized))


def build_indices_per_parameter_point(
    parameter_matrix: np.ndarray,
    parameter_points: list[tuple[float, ...]],
    max_events_per_parameter: int,
    random_seed: int,
) -> dict[tuple[float, ...], np.ndarray]:
    np.random.seed(random_seed)
    normalized_points = np.asarray(
        [normalize_parameter_point(point) for point in np.asarray(parameter_matrix, dtype=float)],
        dtype=float,
    )

    out = {}
    for point in parameter_points:
        point_array = np.asarray(point, dtype=float)
        indices = np.argwhere(np.all(np.isclose(normalized_points, point_array, atol=1e-3), axis=1)).flatten()
        if max_events_per_parameter == -1:
            out[point] = indices
            continue
        elif len(indices) > max_events_per_parameter:
            indices = np.random.choice(indices, max_events_per_parameter, replace=False)
        out[point] = indices
    return out


def structure_data(
    data_file: h5py.File,
    labels_dataset: np.ndarray,
    parameter_points: list[tuple[float, ...]],
    observables_config: list[dict],
    parameter_specs: list[dict],
    weight_spec: dict,
    training_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[float, ...], int], dict[str, int]]:
    ordered_features = []
    feature_index_map = {}

    for obs in observables_config:
        group_name, variable_name = parse_feature_spec(obs)
        values = get_feature_from_h5(data_file, group_name, variable_name)[training_indices]
        feature_index_map[feature_key(group_name, variable_name)] = len(ordered_features)
        ordered_features.append(values.reshape(-1, 1))

    parameter_columns = []
    for parameter_spec in parameter_specs:
        param_group, param_variable = parse_feature_spec(parameter_spec)
        parameter_values = get_feature_from_h5(data_file, param_group, param_variable)[training_indices]
        feature_index_map[feature_key(param_group, param_variable)] = len(ordered_features)
        ordered_features.append(parameter_values.reshape(-1, 1))
        parameter_columns.append(parameter_values.reshape(-1, 1))

    weight_group, weight_variable = parse_feature_spec(weight_spec)
    weight_values = get_feature_from_h5(data_file, weight_group, weight_variable)[training_indices]
    feature_index_map[feature_key(weight_group, weight_variable)] = len(ordered_features)
    ordered_features.append(weight_values.reshape(-1, 1))

    X = np.concatenate(ordered_features, axis=1)
    y = labels_dataset[training_indices]

    parameter_matrix = np.concatenate(parameter_columns, axis=1)
    parameter_point_to_category = {point: i for i, point in enumerate(parameter_points)}
    logger.debug("Parameter point to category mapping: %s", parameter_point_to_category)
    y_category = np.asarray(
        [parameter_point_to_category[normalize_parameter_point(point)] for point in parameter_matrix],
        dtype=int,
    )

    y_full = np.concatenate([y.reshape(-1, 1), y_category.reshape(-1, 1)], axis=1)
    return X, y_full, parameter_point_to_category, feature_index_map


def augment_data_for_background(
    X: np.ndarray,
    y: np.ndarray,
    training_parameter_grid: list[tuple[float, ...]],
    parameter_column_indices: list[int],
    parameter_point_to_category: dict[tuple[float, ...], int],
) -> tuple[np.ndarray, np.ndarray]:
    background_indices = np.argwhere(y[:, 0] == 0).flatten()
    logger.debug("Augmenting data for background: found %d background samples", len(background_indices))
    augmented_X = []
    augmented_y = []
    logger.debug("Augmenting data with training parameter grid: %s", training_parameter_grid)

    for idx in background_indices:
        doing_first_parameter_point = True
        for parameter_point in training_parameter_grid:
            parameter_category = parameter_point_to_category[parameter_point]
            # We want to replace the default parameter point with the first training parameter point.
            if doing_first_parameter_point:
                X[idx, parameter_column_indices] = np.asarray(parameter_point, dtype=float)
                y[idx, 1] = parameter_category
                doing_first_parameter_point = False
            else: # For all the others we create a new data point with the same features but the new parameter point.
                new_x = X[idx].copy()
                new_x[parameter_column_indices] = np.asarray(parameter_point, dtype=float)
                augmented_X.append(new_x)
                new_y = y[idx].copy()
                new_y[1] = parameter_category
                augmented_y.append(new_y)

    if len(augmented_X) == 0:
        return X, y

    X_ag = np.concatenate([X, np.asarray(augmented_X)], axis=0)
    y_ag = np.concatenate([y, np.asarray(augmented_y)], axis=0)
    logger.debug("Augmented data shape: X=%s, y=%s", X_ag.shape, y_ag.shape)

    return X_ag, y_ag


def norm_weights_per_category_and_sign(weights: np.ndarray, categories: np.ndarray) -> np.ndarray:
    out = weights.copy()
    unique_categories = np.unique(categories)

    for category in unique_categories:
        category_mask = categories == category
        if category == 0:
            denom = out[category_mask].sum()
            if denom != 0.0:
                out[category_mask] = out[category_mask] / denom
            continue

        pos_mask = (out >= 0) & category_mask
        neg_mask = (out < 0) & category_mask

        pos_sum = out[pos_mask].sum()
        neg_sum = np.abs(out[neg_mask]).sum()

        if pos_sum != 0.0:
            out[pos_mask] = out[pos_mask] / pos_sum
        if neg_sum != 0.0:
            out[neg_mask] = out[neg_mask] / neg_sum

    return out


def holdout_mask_from_parameter_matrix(
    parameter_matrix: np.ndarray,
    parameter_specs: list[dict],
    add_to_holdout: list[tuple[float, ...]] | None = None,
    remove_from_holdout: list[tuple[float, ...]] | None = None,
) -> np.ndarray:
    parameter_matrix = np.asarray(parameter_matrix, dtype=float)
    if parameter_matrix.ndim == 1:
        parameter_matrix = parameter_matrix.reshape(-1, 1)

    holdout_mask = np.zeros(parameter_matrix.shape[0], dtype=bool)
    for column_index, spec in enumerate(parameter_specs):
        holdout_values = [normalize_parameter_value(value) for value in spec.get("values_for_testing", [])]
        if len(holdout_values) == 0:
            continue
        holdout_mask |= np.isin(
            np.round(parameter_matrix[:, column_index], PARAMETER_DECIMALS),
            holdout_values,
        )

    # Add specific parameter-point tuples to holdout.
    if add_to_holdout:
        rounded = np.round(parameter_matrix, PARAMETER_DECIMALS)
        for point in add_to_holdout:
            norm_point = np.asarray(normalize_parameter_point(point), dtype=float)
            match = np.all(np.isclose(rounded, norm_point, atol=1e-3), axis=1)
            holdout_mask |= match

    # Remove specific parameter-point tuples from holdout.
    if remove_from_holdout:
        rounded = np.round(parameter_matrix, PARAMETER_DECIMALS)
        for point in remove_from_holdout:
            norm_point = np.asarray(normalize_parameter_point(point), dtype=float)
            match = np.all(np.isclose(rounded, norm_point, atol=1e-3), axis=1)
            holdout_mask &= ~match

    # Print the holdout parameter points for debugging
    holdout_points = parameter_matrix[holdout_mask]
    holdout_points = set(tuple(point) for point in holdout_points)
    for point in holdout_points:
        logger.info("Holdout parameter point: %s", point)

    return holdout_mask


def parameter_point_label(parameter_names: list[str], point: tuple[float, ...]) -> str:
    normalized_point = normalize_parameter_point(point)
    if len(parameter_names) == 1:
        return f"{parameter_names[0]}={normalized_point[0]:g}"
    return ", ".join(f"{name}={value:g}" for name, value in zip(parameter_names, normalized_point))


def parameter_point_slug(parameter_names: list[str], point: tuple[float, ...]) -> str:
    normalized_point = normalize_parameter_point(point)
    parts = []
    for name, value in zip(parameter_names, normalized_point):
        encoded = f"{value:g}".replace("-", "m").replace(".", "p")
        parts.append(f"{name}_{encoded}")
    return "__".join(parts)
