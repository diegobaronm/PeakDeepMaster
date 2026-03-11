import h5py
import numpy as np

def parse_feature_spec(spec: dict) -> tuple[str, str]:
    ignored_keys = {"transformation", "values_for_training", "values_for_testing"}
    for key, value in spec.items():
        if key in ignored_keys:
            continue
        return str(key), str(value)
    raise ValueError(f"Invalid feature spec: {spec}")


def normalize_feature_specs(specs) -> list[dict]:
    return [dict(spec) for spec in specs]


def feature_key(group_name: str, variable_name: str) -> str:
    return f"{group_name}:{variable_name}"


def get_feature_from_h5(data_file: h5py.File, group_name: str, variable_name: str) -> np.ndarray:
    return data_file["INPUTS"][group_name][variable_name][:]


def get_unique_parameters(data: np.ndarray) -> list[float]:
    return [round(float(elem), 1) for elem in sorted(set(data))]


def build_indices_per_parameter(
    parameters_array: np.ndarray,
    parameter_values: list[float],
    max_events_per_parameter: int,
    random_seed: int,
) -> dict[float, np.ndarray]:
    np.random.seed(random_seed)
    out = {}
    for parameter in parameter_values:
        indices = np.argwhere(np.abs(parameters_array - parameter) < 1e-3).flatten()
        if len(indices) > max_events_per_parameter:
            indices = np.random.choice(indices, max_events_per_parameter, replace=False)
        out[parameter] = indices
    return out


def structure_data(
    data_file: h5py.File,
    labels_dataset: np.ndarray,
    parameter_values: list[float],
    parameter_values_for_training: list[float],
    observables_config: list[dict],
    parameter_spec: dict,
    weight_spec: dict,
    training_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[float, int], dict[str, int]]:
    ordered_features = []
    feature_index_map = {}

    for obs in observables_config:
        group_name, variable_name = parse_feature_spec(obs)
        values = get_feature_from_h5(data_file, group_name, variable_name)[training_indices]
        feature_index_map[feature_key(group_name, variable_name)] = len(ordered_features)
        ordered_features.append(values.reshape(-1, 1))

    param_group, param_variable = parse_feature_spec(parameter_spec)
    weight_group, weight_variable = parse_feature_spec(weight_spec)

    X_parameters = get_feature_from_h5(data_file, param_group, param_variable)[training_indices]
    X_weights = get_feature_from_h5(data_file, weight_group, weight_variable)[training_indices]

    feature_index_map[feature_key(param_group, param_variable)] = len(ordered_features)
    ordered_features.append(X_parameters.reshape(-1, 1))
    feature_index_map[feature_key(weight_group, weight_variable)] = len(ordered_features)
    ordered_features.append(X_weights.reshape(-1, 1))

    X = np.concatenate(ordered_features, axis=1)
    y = labels_dataset

    y = y[training_indices]

    y_category = np.empty_like(y)
    parameters_category_dict = {float(c): i for i, c in enumerate(parameter_values)}

    for i, parameter_value in enumerate(X_parameters):
        category = parameters_category_dict[round(float(parameter_value), 1)]
        y_category[i] = category
        if category == 0:
            X_parameters[i] = parameter_values_for_training[0]

    X[:, feature_index_map[feature_key(param_group, param_variable)]] = X_parameters

    y_full = np.concatenate([y.reshape(-1, 1), y_category.reshape(-1, 1)], axis=1)
    return X, y_full, parameters_category_dict, feature_index_map


def augment_data_for_background(
    X: np.ndarray,
    y: np.ndarray,
    parameter_values_for_training: list[float],
    parameter_column_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    background_indices = np.argwhere(y[:, 0] == 0).flatten()
    augmented_X = []
    augmented_y = []

    for idx in background_indices:
        for parameter in parameter_values_for_training[1:]:
            new_x = X[idx].copy()
            new_x[parameter_column_index] = parameter
            augmented_X.append(new_x)
            augmented_y.append(y[idx])

    if len(augmented_X) == 0:
        return X, y

    X_ag = np.concatenate([X, np.asarray(augmented_X)], axis=0)
    y_ag = np.concatenate([y, np.asarray(augmented_y)], axis=0)
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


def holdout_mask_from_parameters(
    y_category: np.ndarray,
    holdout_parameters: list[float],
    parameters_category_dict: dict[float, int],
) -> np.ndarray:
    holdout_categories = [parameters_category_dict[c] for c in holdout_parameters if c in parameters_category_dict]
    if len(holdout_categories) == 0:
        return np.zeros_like(y_category, dtype=bool)
    return np.isin(y_category, holdout_categories)
