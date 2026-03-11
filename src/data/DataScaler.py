import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data.DataHelpers import parse_feature_spec


class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, offset: float = 1e-3):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)


def get_transformer_by_name(name: str):
    normalized = name.strip().lower()
    if normalized == "logscaler":
        return LogScaler()
    if normalized == "standardscaler":
        return StandardScaler()
    if normalized == "minmaxscaler":
        return MinMaxScaler()
    raise ValueError(f"Unsupported transformation '{name}'")


def build_feature_scaler(
    observables_config: list[dict],
    parameter_spec: dict,
    weight_spec: dict,
) -> tuple[ColumnTransformer, list[str]]:
    transformers = []
    ordered_feature_keys = []
    column_idx = 0

    for obs in observables_config:
        group_name, variable_name = parse_feature_spec(obs)
        transformation = obs.get("transformation", "StandardScaler")
        name = f"obs_{variable_name}"
        transformers.append((name, get_transformer_by_name(transformation), [column_idx]))
        ordered_feature_keys.append(f"{group_name}:{variable_name}")
        column_idx += 1

    param_group, param_variable = parse_feature_spec(parameter_spec)
    param_name = f"param_{param_variable}"
    param_transformation = parameter_spec.get("transformation", "MinMaxScaler")
    transformers.append((param_name, get_transformer_by_name(param_transformation), [column_idx]))
    ordered_feature_keys.append(f"{param_group}:{param_variable}")
    column_idx += 1

    weight_group, weight_variable = parse_feature_spec(weight_spec)
    weight_name = f"weight_{weight_variable}"
    transformers.append((weight_name, "passthrough", [column_idx]))
    ordered_feature_keys.append(f"{weight_group}:{weight_variable}")

    return ColumnTransformer(transformers, remainder="drop"), ordered_feature_keys
