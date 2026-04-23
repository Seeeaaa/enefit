import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder to handle numpy data types during JSON serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def format_date(ts) -> str:
    """
    Format pandas timestamp to string dropping hours:minutes:seconds for
    clean file paths.
    """
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d")
    return str(ts)


def get_model_and_meta_paths(
    models_dir: str, notebook: str, split: dict, purpose: str, model_name: str
) -> tuple[Path, Path, Path]:
    """
    Constructs clean directory paths and returns (model_path, meta_path,
    history_path).
    """
    train_bounds = (
        f"{format_date(split['train'][0])}_{format_date(split['train'][1])}"
    )
    model_dir = (
        Path(models_dir) / notebook / purpose / train_bounds / model_name
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "xgb": model_dir / "model.ubj",
        "xgb_booster": model_dir / "model.ubj",
        "lgbm": model_dir / "model.pkl",
        "cb": model_dir / "model.cbm",
    }
    if model_name not in paths:
        raise ValueError(f"Unknown model_name: {model_name}")

    return (
        paths[model_name],
        model_dir / "meta.json",
        model_dir / "history.json",
    )


def save_model_unified(model_name: str, model, model_path: Path):
    """
    Persists model weights to disk using framework-specific serializers:
    - XGBoost / CatBoost: native .save_model() for binary compatibility
    - LightGBM: Joblib (pickle) to preserve the Scikit-Learn wrapper
    state.
    """
    if model_name in ["xgb", "xgb_booster", "cb"]:
        model.save_model(str(model_path))
    elif model_name == "lgbm":
        joblib.dump(model, model_path)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def load_model_unified(
    model_name: str, model_cls, model_params: dict, model_path: Path
):
    """Unified load behavior for different model types."""
    if model_name == "xgb":
        model = model_cls(**model_params)
        model.load_model(str(model_path))
        return model
    elif model_name == "xgb_booster":
        booster = model_cls()
        booster.load_model(str(model_path))
        return booster
    elif model_name == "lgbm":
        return joblib.load(model_path)
    elif model_name == "cb":
        model = model_cls(**model_params)
        model.load_model(str(model_path))
        return model
    raise ValueError(f"Unknown model_name: {model_name}")


def load_cache_meta(
    meta_path: Path, split: dict, cache_params: dict
) -> dict | None:
    """
    Validates if a cached model exists and matches the current run.

    Normalizes 'cache_params' to standard Python types (removing
    numpy-specific types) to ensure a reliable comparison with the
    parameters stored on disk.

    Returns the meta dictionary on a match, otherwise returns None.
    """
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # serialize and deserialize cache_params to guarantee identical
        # structures
        normalized_params = json.loads(
            json.dumps(cache_params, cls=NumpyEncoder)
        )

        if (
            meta.get("train_start") == str(split["train"][0])
            and meta.get("train_end") == str(split["train"][1])
            and meta.get("test_start") == str(split["test"][0])
            and meta.get("test_end") == str(split["test"][1])
            and meta.get("model_params") == normalized_params
        ):
            return meta
    except Exception:
        pass
    return None


def save_cache_meta(meta_path: Path, split: dict, cache_params: dict):
    """Saves the metadata explicitly tracking parameters and splits."""
    meta = {
        "train_start": str(split["train"][0]),
        "train_end": str(split["train"][1]),
        "test_start": str(split["test"][0]),
        "test_end": str(split["test"][1]),
        "model_params": cache_params,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_or_train_sklearn(
    models_dir: str,
    notebook: str,
    purpose: str,
    model_name: str,
    model_cls,
    model_params: dict,
    split: dict,
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: list = ["datetime"],
    cat_cols: list | None = None,
    eval_sample_size: int = 100_000,
    random_state: int = 10,
) -> tuple[object, bool, dict | None]:
    """
    Pipeline for Scikit-Learn API wrappers. 
    
    Implements 'lazy loading': if a valid cache is found, data
    preparation (splitting/sampling) is skipped entirely. Handles
    temporal splitting internally and ensures strict validation by
    removing 'eval_set' rows from the training pool.
    """
    model_path, meta_path, history_path = get_model_and_meta_paths(
        models_dir, notebook, split, purpose, model_name
    )

    cache_params = {
        "model_params": model_params,
        "eval_sample_size": eval_sample_size,
        "random_state": random_state,
        "drop_cols": sorted(drop_cols) if drop_cols else [],
    }

    meta = load_cache_meta(meta_path, split, cache_params)
    if model_path.exists() and meta is not None:
        try:
            model = load_model_unified(
                model_name, model_cls, model_params, model_path
            )
            history = None
            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as f:
                    history = json.load(f)
            return model, False, history
        except Exception:
            pass

    if df is None:
        raise ValueError(
            f"Cache miss for {model_name}, but df was not provided."
        )

    mask = (df["datetime"] >= split["train"][0]) & (
        df["datetime"] <= split["train"][1]
    )
    train_df = df.loc[mask, ~df.columns.isin(drop_cols)]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # optional eval_set logic; strictly None if size is 0
    eval_set = None
    if eval_sample_size > 0:
        n_samples = min(eval_sample_size, len(X_train))
        eval_idx = X_train.sample(n_samples, random_state=random_state).index
        
        # prepare evaluation set
        eval_set = [(X_train.loc[eval_idx], y_train.loc[eval_idx])]
        
        # remove eval rows from training data to prevent leakage
        X_train = X_train.drop(index=eval_idx)
        y_train = y_train.drop(index=eval_idx)

    model = model_cls(**model_params)
    history = None

    # use kwargs to avoid passing eval_set=None
    fit_kwargs = {}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    if model_name == "xgb":
        model.fit(X_train, y_train, verbose=0, **fit_kwargs)
        if eval_set is not None:
            history = model.evals_result()
    elif model_name == "lgbm":
        model.fit(X_train, y_train, categorical_feature=cat_cols, **fit_kwargs)
        if eval_set is not None:
            history = getattr(model, "evals_result_", None)
    elif model_name == "cb":
        model.fit(X_train, y_train, cat_features=cat_cols, **fit_kwargs)
        if eval_set is not None:
            history = getattr(model, "evals_result_", None)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    save_model_unified(model_name, model, model_path)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    save_cache_meta(meta_path, split, cache_params)

    return model, True, history


def load_or_train_core(
    models_dir: str,
    notebook: str,
    purpose: str,
    split: dict,
    model_params: dict,
    df: pd.DataFrame,
    target_col: str = "target",
    drop_cols: list = ["datetime"],
    eval_sample_size: int = 100_000,
    random_state: int = 10,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: int | bool = 100,
) -> tuple[xgb.Booster, bool, dict | None]:
    """
    Pipeline for XGBoost Core API. Handles internal DMatrix creation and
    temporal splitting. Returns (booster, was_trained, train_history).
    """
    # xgb_booster for loading/saving logic
    internal_model_name = "xgb_booster"
    model_path, meta_path, history_path = get_model_and_meta_paths(
        models_dir, notebook, split, purpose, internal_model_name
    )

    # training parameters in cache check
    cache_params = {
        "model_params": model_params,
        "eval_sample_size": eval_sample_size,
        "random_state": random_state,
        "drop_cols": sorted(drop_cols) if drop_cols else [],
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
    }

    meta = load_cache_meta(meta_path, split, cache_params)
    if model_path.exists() and meta is not None:
        try:
            booster = load_model_unified(
                internal_model_name, xgb.Booster, {}, model_path
            )
            history = None
            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as f:
                    history = json.load(f)
            return booster, False, history
        except Exception:
            pass

    if df is None:
        raise ValueError(
            f"Cache miss for {internal_model_name}, but df was not provided."
        )

    # splitting
    mask = (df["datetime"] >= split["train"][0]) & (
        df["datetime"] <= split["train"][1]
    )
    train_df = df.loc[mask, ~df.columns.isin(drop_cols)]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # DMatrix creation and optional evaluation set
    if eval_sample_size > 0:
        n_samples = min(eval_sample_size, len(X_train))
        eval_idx = X_train.sample(n_samples, random_state=random_state).index
        
        # separate DMatrices to ensure eval is excluded from train
        dtrain = xgb.DMatrix(
            X_train.drop(index=eval_idx), 
            label=y_train.drop(index=eval_idx), 
            enable_categorical=True
        )
        deval = xgb.DMatrix(
            X_train.loc[eval_idx],
            label=y_train.loc[eval_idx],
            enable_categorical=True,
        )
        evals = [(dtrain, "train"), (deval, "eval")]
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        evals = [(dtrain, "train")]

    # training
    history = {}
    booster = xgb.train(
        params=model_params,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        evals_result=history,
    )

    save_model_unified(internal_model_name, booster, model_path)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    save_cache_meta(meta_path, split, cache_params)

    return booster, True, history
