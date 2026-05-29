from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older sklearn fallback
    FrozenEstimator = None


DATA_PATH = Path("UFC_full_data_silver.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "ufc_advanced_calibrated.joblib"
RESULTS_PATH = ARTIFACT_DIR / "ufc_advanced_pipeline_results.json"

RANDOM_STATE = 42
ROUND_SECONDS = 5 * 60

FIGHT_STATS = [
    "knockdowns",
    "total_strikes_att",
    "total_strikes_succ",
    "sig_strikes_att",
    "sig_strikes_succ",
    "takedown_att",
    "takedown_succ",
    "submission_att",
    "reversals",
    "ctrl_time_sec",
]

POSITION_STATS = [
    "head_att",
    "head_succ",
    "body_att",
    "body_succ",
    "leg_att",
    "leg_succ",
    "distance_att",
    "distance_succ",
    "clinch_att",
    "clinch_succ",
    "ground_att",
    "ground_succ",
]

BASE_FIGHTER_FEATURES = [
    "age_years",
    "height_cm",
    "reach_cm",
]

HISTORY_FEATURES = [
    "prior_fights",
    "has_prior_fight",
    "has_3_prior_fights",
    "days_since_last_fight",
    "days_since_last_fight_missing",
    "last_fight_duration_min",
    "last_fight_win",
    "last_fight_loss",
    "last_fight_ko_loss",
    "recent_1_sig_landed_per_min",
    "recent_1_sig_absorbed_per_min",
    "recent_1_takedowns_landed_per_15",
    "recent_1_takedowns_attempted_per_15",
    "recent_1_submissions_attempted_per_15",
    "recent_1_control_minutes_per_15",
    "recent_1_knockdowns_per_15",
    "recent_1_striking_defense",
    "recent_3_win_rate",
    "recent_3_ko_loss_rate",
    "recent_3_fight_duration_min",
    "recent_3_sig_landed_per_min",
    "recent_3_sig_absorbed_per_min",
    "recent_3_takedowns_landed_per_15",
    "recent_3_takedowns_attempted_per_15",
    "recent_3_submissions_attempted_per_15",
    "recent_3_control_minutes_per_15",
    "recent_3_knockdowns_per_15",
    "recent_3_striking_defense",
    "recent_5_win_rate",
    "recent_5_ko_loss_rate",
    "recent_5_fight_duration_min",
    "recent_5_sig_landed_per_min",
    "recent_5_sig_absorbed_per_min",
    "recent_5_takedowns_landed_per_15",
    "recent_5_takedowns_attempted_per_15",
    "recent_5_submissions_attempted_per_15",
    "recent_5_control_minutes_per_15",
    "recent_5_knockdowns_per_15",
    "recent_5_striking_defense",
    "ewma_win_rate",
    "ewma_ko_loss_rate",
    "ewma_sig_landed_per_min",
    "ewma_sig_absorbed_per_min",
    "ewma_takedowns_landed_per_15",
    "ewma_control_minutes_per_15",
    "ewma_striking_defense",
]

FIGHTER_FEATURES = BASE_FIGHTER_FEATURES + HISTORY_FEATURES

DIFF_FEATURE_COLUMNS = [f"{feature}_diff" for feature in FIGHTER_FEATURES]
FEATURE_COLUMNS = [
    *DIFF_FEATURE_COLUMNS,
    "same_stance",
    "title_fight",
    "scheduled_rounds",
    "female_division",
]

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fighter_id(df: pd.DataFrame, side: int) -> pd.Series:
    profile_url = df[f"f_{side}_fighter_url"].fillna("").astype(str)
    fight_url = df[f"f_{side}_url"].fillna("").astype(str)
    name = df[f"f_{side}_name"].fillna("").astype(str)
    return np.select(
        [profile_url.ne(""), fight_url.ne("")],
        [profile_url, fight_url],
        default=name,
    )


def parse_finish_time_seconds(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if ":" not in text:
        return pd.to_numeric(text, errors="coerce")
    minutes, seconds = text.split(":", 1)
    try:
        return float(minutes) * 60.0 + float(seconds)
    except ValueError:
        return np.nan


def fight_duration_seconds(df: pd.DataFrame) -> pd.Series:
    finish_round = numeric(df["finish_round"]).fillna(numeric(df["num_rounds"]))
    seconds_in_round = df["finish_time"].map(parse_finish_time_seconds)
    scheduled_duration = numeric(df["num_rounds"]).fillna(3) * ROUND_SECONDS
    duration = (finish_round - 1).clip(lower=0) * ROUND_SECONDS + seconds_in_round
    return duration.fillna(scheduled_duration).clip(lower=0, upper=scheduled_duration)


def result_contains(df: pd.DataFrame, pattern: str) -> pd.Series:
    return df["result"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def is_decision(df: pd.DataFrame) -> pd.Series:
    return result_contains(df, r"\bdec|decision")


def is_ko_tko(df: pd.DataFrame) -> pd.Series:
    return result_contains(df, r"ko|tko|doctor")


def load_clean_fights(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    valid_target = df["winner"].eq(df["f_1_name"]) | df["winner"].eq(df["f_2_name"])
    df = df.loc[valid_target & df["event_date"].notna()].copy()
    df = df.sort_values(["event_date", "event_name", "fight_url"]).reset_index(drop=True)
    df["target_f1_win"] = df["winner"].eq(df["f_1_name"]).astype(int)
    df["fight_duration_sec"] = fight_duration_seconds(df)
    df["f_1_id"] = fighter_id(df, 1)
    df["f_2_id"] = fighter_id(df, 2)
    return df


def round_stat_sum(df: pd.DataFrame, side: int, stat: str) -> pd.Series:
    values = pd.Series(0.0, index=df.index)
    for round_number in range(1, 6):
        col = f"f_{side}_r{round_number}_{stat}"
        if col in df.columns:
            values = values + numeric(df[col]).fillna(0.0)
    return values


def make_long_history(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    ko = is_ko_tko(df)
    decision = is_decision(df)
    for side, opp in [(1, 2), (2, 1)]:
        won = df["winner"].eq(df[f"f_{side}_name"])
        lost = df["winner"].eq(df[f"f_{opp}_name"])
        side_data = pd.DataFrame(
            {
                "event_date": df["event_date"],
                "fight_url": df["fight_url"],
                "fighter_id": df[f"f_{side}_id"],
                "win": won.astype(float),
                "loss": lost.astype(float),
                "finish_win": (won & ~decision).astype(float),
                "finish_loss": (lost & ~decision).astype(float),
                "ko_loss": (lost & ko).astype(float),
                "fight_duration_sec": df["fight_duration_sec"],
            }
        )

        for stat in FIGHT_STATS:
            side_data[stat] = numeric(df[f"f_{side}_{stat}"]).fillna(0.0)
            side_data[f"opp_{stat}"] = numeric(df[f"f_{opp}_{stat}"]).fillna(0.0)

        for stat in POSITION_STATS:
            side_data[stat] = round_stat_sum(df, side, stat)

        frames.append(side_data)

    return pd.concat(frames, ignore_index=True)


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    long_history = make_long_history(df)
    date_level = (
        long_history.groupby(["fighter_id", "event_date"], as_index=False)
        .agg(
            fight_count=("fight_url", "count"),
            win=("win", "mean"),
            loss=("loss", "mean"),
            ko_loss=("ko_loss", "mean"),
            fight_duration_sec=("fight_duration_sec", "mean"),
            sig_strikes_succ=("sig_strikes_succ", "sum"),
            sig_strikes_att=("sig_strikes_att", "sum"),
            opp_sig_strikes_succ=("opp_sig_strikes_succ", "sum"),
            opp_sig_strikes_att=("opp_sig_strikes_att", "sum"),
            takedown_succ=("takedown_succ", "sum"),
            takedown_att=("takedown_att", "sum"),
            submission_att=("submission_att", "sum"),
            ctrl_time_sec=("ctrl_time_sec", "sum"),
            knockdowns=("knockdowns", "sum"),
        )
        .sort_values(["fighter_id", "event_date"])
    )

    grouped = date_level.groupby("fighter_id", group_keys=False)
    previous_event_date = grouped["event_date"].shift(1)
    date_level["prior_fights"] = grouped["fight_count"].cumsum() - date_level["fight_count"]
    date_level["has_prior_fight"] = date_level["prior_fights"].ge(1).astype(float)
    date_level["has_3_prior_fights"] = date_level["prior_fights"].ge(3).astype(float)
    date_level["days_since_last_fight"] = (date_level["event_date"] - previous_event_date).dt.days
    date_level["days_since_last_fight_missing"] = date_level["days_since_last_fight"].isna().astype(float)
    date_level["last_fight_duration_min"] = grouped["fight_duration_sec"].shift(1) / 60.0
    date_level["last_fight_win"] = grouped["win"].shift(1)
    date_level["last_fight_loss"] = grouped["loss"].shift(1)
    date_level["last_fight_ko_loss"] = grouped["ko_loss"].shift(1)

    duration_min = (date_level["fight_duration_sec"] / 60.0).replace(0, np.nan)
    duration_15 = (date_level["fight_duration_sec"] / (15.0 * 60.0)).replace(0, np.nan)
    date_level["fight_duration_min"] = duration_min
    date_level["sig_landed_per_min"] = safe_rate(date_level["sig_strikes_succ"], duration_min)
    date_level["sig_absorbed_per_min"] = safe_rate(date_level["opp_sig_strikes_succ"], duration_min)
    date_level["takedowns_landed_per_15"] = safe_rate(date_level["takedown_succ"], duration_15)
    date_level["takedowns_attempted_per_15"] = safe_rate(date_level["takedown_att"], duration_15)
    date_level["submissions_attempted_per_15"] = safe_rate(date_level["submission_att"], duration_15)
    date_level["control_minutes_per_15"] = safe_rate(date_level["ctrl_time_sec"] / 60.0, duration_15)
    date_level["knockdowns_per_15"] = safe_rate(date_level["knockdowns"], duration_15)
    date_level["striking_defense"] = 1.0 - safe_rate(
        date_level["opp_sig_strikes_succ"],
        date_level["opp_sig_strikes_att"],
    )

    recent_source_columns = {
        "win_rate": "win",
        "ko_loss_rate": "ko_loss",
        "fight_duration_min": "fight_duration_min",
        "sig_landed_per_min": "sig_landed_per_min",
        "sig_absorbed_per_min": "sig_absorbed_per_min",
        "takedowns_landed_per_15": "takedowns_landed_per_15",
        "takedowns_attempted_per_15": "takedowns_attempted_per_15",
        "submissions_attempted_per_15": "submissions_attempted_per_15",
        "control_minutes_per_15": "control_minutes_per_15",
        "knockdowns_per_15": "knockdowns_per_15",
        "striking_defense": "striking_defense",
    }

    for window in [1, 3, 5]:
        for out_name, source_col in recent_source_columns.items():
            date_level[f"recent_{window}_{out_name}"] = grouped[source_col].transform(
                lambda values, w=window: values.shift(1).rolling(w, min_periods=1).mean()
            )

    ewma_source_columns = {
        "win_rate": "win",
        "ko_loss_rate": "ko_loss",
        "sig_landed_per_min": "sig_landed_per_min",
        "sig_absorbed_per_min": "sig_absorbed_per_min",
        "takedowns_landed_per_15": "takedowns_landed_per_15",
        "control_minutes_per_15": "control_minutes_per_15",
        "striking_defense": "striking_defense",
    }
    for out_name, source_col in ewma_source_columns.items():
        date_level[f"ewma_{out_name}"] = grouped[source_col].transform(
            lambda values: values.shift(1).ewm(alpha=0.45, adjust=False, min_periods=1).mean()
        )

    return date_level[["fighter_id", "event_date", *HISTORY_FEATURES]]


def add_side_history(base: pd.DataFrame, history: pd.DataFrame, side: int) -> pd.DataFrame:
    history_cols = [col for col in history.columns if col not in {"fighter_id", "event_date"}]
    renamed = history.rename(columns={col: f"f_{side}_hist_{col}" for col in history_cols})
    return base.merge(
        renamed,
        how="left",
        left_on=[f"f_{side}_id", "event_date"],
        right_on=["fighter_id", "event_date"],
    ).drop(columns=["fighter_id"])


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_side_features(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for side in [1, 2]:
        prefix = f"f_{side}"
        hist = f"{prefix}_hist"

        dob = pd.to_datetime(base[f"{prefix}_fighter_dob"], errors="coerce")
        base[f"{prefix}_age_years"] = (base["event_date"] - dob).dt.days / 365.25
        base[f"{prefix}_height_cm"] = numeric(base[f"{prefix}_fighter_height_cm"])
        base[f"{prefix}_reach_cm"] = numeric(base[f"{prefix}_fighter_reach_cm"])
        for feature in HISTORY_FEATURES:
            base[f"{prefix}_{feature}"] = base[f"{hist}_{feature}"]

    return base


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    history = build_historical_features(df)
    base = add_side_history(df, history, 1)
    base = add_side_history(base, history, 2)
    base = add_side_features(base)

    frame = pd.DataFrame(
        {
            "event_date": base["event_date"],
            "event_name": base["event_name"],
            "fight_url": base["fight_url"],
            "f_1_name": base["f_1_name"],
            "f_2_name": base["f_2_name"],
            "winner": base["winner"],
            "target_f1_win": base["target_f1_win"],
            "same_stance": (
                base["f_1_fighter_stance"].fillna("").astype(str)
                == base["f_2_fighter_stance"].fillna("").astype(str)
            ).astype(int),
            "title_fight": base["title_fight"].astype(bool).astype(int),
            "scheduled_rounds": numeric(base["num_rounds"]),
            "female_division": base["gender"].fillna("").astype(str).eq("F").astype(int),
        }
    )

    for feature in FIGHTER_FEATURES:
        frame[f"{feature}_diff"] = base[f"f_1_{feature}"] - base[f"f_2_{feature}"]

    return frame


def chronological_split(frame: pd.DataFrame, train_date_fraction: float = 0.8):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * train_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    train_mask = frame["event_date"].le(cutoff_date)
    test_mask = frame["event_date"].gt(cutoff_date)
    return train_mask, test_mask, cutoff_date


def temporal_train_calibration_split(frame: pd.DataFrame, fit_date_fraction: float = 0.85):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * fit_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    fit_mask = frame["event_date"].le(cutoff_date)
    calibration_mask = frame["event_date"].gt(cutoff_date)
    return fit_mask, calibration_mask, cutoff_date


def symmetric_training_set(X: pd.DataFrame, y: pd.Series):
    swapped = X.copy()
    swapped[DIFF_FEATURE_COLUMNS] = -swapped[DIFF_FEATURE_COLUMNS]
    return pd.concat([X, swapped], ignore_index=True), pd.concat(
        [y.reset_index(drop=True), 1 - y.reset_index(drop=True)], ignore_index=True
    )


def metric_block(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def make_pipeline(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=4000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
            **params,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    elif model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **params,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])


def default_params(model_name: str) -> dict[str, Any]:
    defaults = {
        "logistic_regression": {"C": 1.0, "class_weight": None},
        "random_forest": {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 18,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "extra_trees": {
            "n_estimators": 500,
            "max_depth": 7,
            "min_samples_leaf": 18,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "hist_gradient_boosting": {
            "max_iter": 300,
            "learning_rate": 0.035,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 35,
            "l2_regularization": 0.1,
        },
    }
    return defaults[model_name]


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 20.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 45),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.45, 0.65, 0.85]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
        }
    raise ValueError(model_name)


def tune_model(
    model_name: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    trials: int,
) -> dict[str, Any]:
    if trials <= 0:
        params = default_params(model_name)
        model = make_pipeline(model_name, params)
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_valid)[:, 1]
        return {
            "best_params": params,
            "validation_metrics": metric_block(y_valid, probabilities),
            "trials": 0,
        }

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        model = make_pipeline(model_name, params)
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_valid)[:, 1]
        return log_loss(y_valid, np.clip(probabilities, 1e-6, 1 - 1e-6))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_model = make_pipeline(model_name, study.best_params)
    best_model.fit(X_fit, y_fit)
    probabilities = best_model.predict_proba(X_valid)[:, 1]
    return {
        "best_params": study.best_params,
        "best_value_log_loss": float(study.best_value),
        "validation_metrics": metric_block(y_valid, probabilities),
        "trials": trials,
    }


def fit_prefit_calibrator(
    base_model: Pipeline,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
            cv=None,
        )
    else:
        calibrator = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit",
        )
    calibrator.fit(X_calibration, y_calibration)
    return calibrator


def calibration_summary(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    prob_true, prob_pred = calibration_curve(
        y_true,
        np.clip(probabilities, 1e-6, 1 - 1e-6),
        n_bins=bins,
        strategy="quantile",
    )
    rows = [
        {
            "mean_predicted_probability": float(pred),
            "observed_win_rate": float(true),
            "gap": float(pred - true),
        }
        for pred, true in zip(prob_pred, prob_true)
    ]
    return {
        "bins": rows,
        "mean_abs_calibration_gap": float(np.mean([abs(row["gap"]) for row in rows])) if rows else np.nan,
    }


def model_importance(model: Pipeline, feature_names: list[str]) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        key = "standardized_coef"
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        key = "feature_importance"
    else:
        return []
    return sorted(
        [{"feature": feature, key: float(value)} for feature, value in zip(feature_names, values)],
        key=lambda item: abs(float(item[key])),
        reverse=True,
    )


def permutation_importance_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_log_loss",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "neg_log_loss_importance_mean": float(mean),
            "neg_log_loss_importance_std": float(std),
        }
        for feature, mean, std in zip(
            X_test.columns,
            result.importances_mean,
            result.importances_std,
        )
    ]
    return sorted(rows, key=lambda row: row["neg_log_loss_importance_mean"], reverse=True)[:top_n]


def run_pipeline(trials: int, calibration_method: str) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame = build_training_frame(fights)
    train_mask, test_mask, test_cutoff = chronological_split(frame, train_date_fraction=0.8)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    fit_mask, validation_mask, validation_cutoff = temporal_train_calibration_split(
        train_frame,
        fit_date_fraction=0.85,
    )

    X_fit, y_fit = symmetric_training_set(
        train_frame.loc[fit_mask, FEATURE_COLUMNS],
        train_frame.loc[fit_mask, "target_f1_win"],
    )
    X_valid, y_valid = symmetric_training_set(
        train_frame.loc[validation_mask, FEATURE_COLUMNS],
        train_frame.loc[validation_mask, "target_f1_win"],
    )
    X_train_sym, y_train_sym = symmetric_training_set(
        train_frame[FEATURE_COLUMNS],
        train_frame["target_f1_win"],
    )
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["target_f1_win"]

    model_results = {}
    for model_name in MODEL_NAMES:
        tuned = tune_model(model_name, X_fit, y_fit, X_valid, y_valid, trials)
        model = make_pipeline(model_name, tuned["best_params"])
        model.fit(X_train_sym, y_train_sym)
        probabilities = model.predict_proba(X_test)[:, 1]
        model_results[model_name] = {
            **tuned,
            "test_metrics_uncalibrated": metric_block(y_test, probabilities),
            "model_importance": model_importance(model, FEATURE_COLUMNS)[:20],
        }

    best_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["validation_metrics"]["log_loss"],
    )[0]
    best_params = model_results[best_model_name]["best_params"]

    train_fit_mask, train_cal_mask, final_calibration_cutoff = temporal_train_calibration_split(
        train_frame,
        fit_date_fraction=0.85,
    )
    X_base_fit, y_base_fit = symmetric_training_set(
        train_frame.loc[train_fit_mask, FEATURE_COLUMNS],
        train_frame.loc[train_fit_mask, "target_f1_win"],
    )
    X_cal, y_cal = symmetric_training_set(
        train_frame.loc[train_cal_mask, FEATURE_COLUMNS],
        train_frame.loc[train_cal_mask, "target_f1_win"],
    )
    base_model = make_pipeline(best_model_name, best_params)
    base_model.fit(X_base_fit, y_base_fit)
    calibrated_model = fit_prefit_calibrator(
        base_model,
        X_cal,
        y_cal,
        method=calibration_method,
    )
    uncalibrated_probabilities = base_model.predict_proba(X_test)[:, 1]
    calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]

    permutation_top = permutation_importance_report(calibrated_model, X_test, y_test)

    full_fit_mask, full_cal_mask, artifact_calibration_cutoff = temporal_train_calibration_split(
        frame,
        fit_date_fraction=0.85,
    )
    X_full_fit, y_full_fit = symmetric_training_set(
        frame.loc[full_fit_mask, FEATURE_COLUMNS],
        frame.loc[full_fit_mask, "target_f1_win"],
    )
    X_full_cal, y_full_cal = symmetric_training_set(
        frame.loc[full_cal_mask, FEATURE_COLUMNS],
        frame.loc[full_cal_mask, "target_f1_win"],
    )
    X_full_sym, _ = symmetric_training_set(frame[FEATURE_COLUMNS], frame["target_f1_win"])
    final_base_model = make_pipeline(best_model_name, best_params)
    final_base_model.fit(X_full_fit, y_full_fit)
    final_model = fit_prefit_calibrator(
        final_base_model,
        X_full_cal,
        y_full_cal,
        method=calibration_method,
    )

    artifact = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "diff_feature_columns": DIFF_FEATURE_COLUMNS,
        "feature_medians": X_full_sym.median(numeric_only=True).to_dict(),
        "metadata": {
            "model_name": best_model_name,
            "best_params": best_params,
            "feature_count": len(FEATURE_COLUMNS),
            "data_path": str(DATA_PATH),
            "dataset_rows_with_target": int(len(frame)),
            "dataset_min_date": fights["event_date"].min().strftime("%Y-%m-%d"),
            "dataset_max_date": fights["event_date"].max().strftime("%Y-%m-%d"),
            "calibration_method": calibration_method,
            "artifact_calibration_cutoff_fit_lte": artifact_calibration_cutoff.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(artifact, MODEL_PATH)

    train_prior = float(train_frame["target_f1_win"].mean())
    side_prior_probs = np.full(shape=len(y_test), fill_value=train_prior)
    coinflip_probs = np.full(shape=len(y_test), fill_value=0.5)

    results = {
        "data_path": str(DATA_PATH),
        "artifact_path": str(MODEL_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "leakage_policy": [
            "Current-fight result, winner, finish fields, aggregate stats and per-round stats are not used directly as row features.",
            "Historical stats use shifted rolling and EWMA windows by event_date, so same-date current fights are excluded.",
            "Last-fight status features are taken from the fighter's previous UFC event only.",
            "Betting odds and ranking snapshots are still excluded because their timestamp in the CSV is not audited here.",
            "Mutable UFCStats career profile performance fields are excluded; only date-safe physical profile fields are used.",
            "Training rows are symmetrically augmented with swapped fighters to reduce f_1/f_2 ordering bias.",
        ],
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "rows_total_with_target": int(len(frame)),
        "train_rows_original": int(len(train_frame)),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "validation_rows_after_symmetric_augmentation": int(len(X_valid)),
        "test_rows": int(len(test_frame)),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "validation_cutoff_fit_lte": validation_cutoff.strftime("%Y-%m-%d"),
        "test_date_min": test_frame["event_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": test_frame["event_date"].max().strftime("%Y-%m-%d"),
        "train_f1_win_rate": train_prior,
        "test_f1_win_rate": float(y_test.mean()),
        "optuna_trials_per_model": int(trials),
        "models": model_results,
        "best_model_by_validation_log_loss": best_model_name,
        "best_model_params": best_params,
        "calibration": {
            "method": calibration_method,
            "cutoff_fit_lte": final_calibration_cutoff.strftime("%Y-%m-%d"),
            "uncalibrated_test_metrics": metric_block(y_test, uncalibrated_probabilities),
            "calibrated_test_metrics": metric_block(y_test, calibrated_probabilities),
            "calibration_curve": calibration_summary(y_test, calibrated_probabilities),
        },
        "permutation_importance_top": permutation_top,
        "baselines": {
            "side_prior_probability": metric_block(y_test, side_prior_probs),
            "coinflip": metric_block(y_test, coinflip_probs),
        },
        "analysis": {
            "best_model": best_model_name,
            "artifact_training_note": (
                "Holdout metrics use only dates through the train cutoff for fitting/calibration. "
                "The saved artifact is refit after evaluation on all available rows with a final "
                "temporal fit/calibration split."
            ),
            "artifact_calibration_cutoff_fit_lte": artifact_calibration_cutoff.strftime("%Y-%m-%d"),
            "calibration_delta_log_loss": float(
                metric_block(y_test, calibrated_probabilities)["log_loss"]
                - metric_block(y_test, uncalibrated_probabilities)["log_loss"]
            ),
            "calibration_delta_brier": float(
                metric_block(y_test, calibrated_probabilities)["brier_score"]
                - metric_block(y_test, uncalibrated_probabilities)["brier_score"]
            ),
            "top_features_by_permutation": [row["feature"] for row in permutation_top[:10]],
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced UFC pre-fight model pipeline with Optuna tuning and calibration."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Optuna trials per model. Use 0 to run default hyperparameters only.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Probability calibration method.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(trials=args.trials, calibration_method=args.calibration_method)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
