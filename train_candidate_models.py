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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.svm import LinearSVC

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover - older sklearn fallback
    FrozenEstimator = None

from ufc_ablation_analysis import (
    ARTIFACT_DIR,
    DATA_PATH,
    build_feature_frame,
    load_clean_fights,
    symmetric_training_set,
    unique_features,
)
from ufc_advanced_pipeline import RANDOM_STATE


RESULTS_PATH = ARTIFACT_DIR / "ufc_candidate_models_results.json"
BEST_CV_MODEL_PATH = ARTIFACT_DIR / "ufc_candidate_best_cv_calibrated.joblib"
BEST_HOLDOUT_MODEL_PATH = ARTIFACT_DIR / "ufc_candidate_best_holdout_calibrated.joblib"

CANDIDATE_GROUPS = [
    "physical",
    "context",
    "cumulative_results",
    "cumulative_style",
    "last_fight_state",
]
MODEL_NAMES = ["logistic_regression", "extra_trees", "linear_svm"]


def candidate_features() -> list[str]:
    return unique_features(CANDIDATE_GROUPS)


def chronological_split(frame: pd.DataFrame, train_date_fraction: float = 0.8):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * train_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    train_mask = frame["event_date"].le(cutoff_date)
    test_mask = frame["event_date"].gt(cutoff_date)
    return train_mask, test_mask, cutoff_date


def temporal_fit_calibration_split(frame: pd.DataFrame, fit_date_fraction: float = 0.85):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * fit_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    fit_mask = frame["event_date"].le(cutoff_date)
    calibration_mask = frame["event_date"].gt(cutoff_date)
    return fit_mask, calibration_mask, cutoff_date


def rolling_temporal_folds(
    frame: pd.DataFrame,
    n_folds: int,
    min_train_fraction: float = 0.45,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.10,
) -> list[dict[str, Any]]:
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    date_count = len(dates)
    min_train = max(20, int(date_count * min_train_fraction))
    cal_size = max(8, int(date_count * calibration_fraction))
    val_size = max(8, int(date_count * validation_fraction))
    available = date_count - min_train - cal_size - val_size
    if available < 0:
        raise ValueError("Not enough event dates to build rolling folds.")

    step = max(1, available // max(1, n_folds - 1))
    folds = []
    for fold_idx in range(n_folds):
        fit_end = min_train + fold_idx * step
        cal_end = fit_end + cal_size
        val_end = cal_end + val_size
        if val_end > date_count:
            break
        fit_dates = dates[:fit_end]
        cal_dates = dates[fit_end:cal_end]
        val_dates = dates[cal_end:val_end]
        folds.append(
            {
                "fold": fold_idx + 1,
                "fit_mask": frame["event_date"].isin(fit_dates),
                "calibration_mask": frame["event_date"].isin(cal_dates),
                "validation_mask": frame["event_date"].isin(val_dates),
                "fit_date_max": pd.Timestamp(fit_dates[-1]).strftime("%Y-%m-%d"),
                "calibration_date_min": pd.Timestamp(cal_dates[0]).strftime("%Y-%m-%d"),
                "calibration_date_max": pd.Timestamp(cal_dates[-1]).strftime("%Y-%m-%d"),
                "validation_date_min": pd.Timestamp(val_dates[0]).strftime("%Y-%m-%d"),
                "validation_date_max": pd.Timestamp(val_dates[-1]).strftime("%Y-%m-%d"),
            }
        )
    return folds


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


def probability_bucket_report(
    y_true: pd.Series,
    probabilities: np.ndarray,
    bucket_edges: list[float] | None = None,
) -> dict[str, Any]:
    if bucket_edges is None:
        bucket_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    rows = []
    for left, right in zip(bucket_edges[:-1], bucket_edges[1:]):
        if right == bucket_edges[-1]:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        rows.append(
            {
                "bucket": f"{left:.1f}-{right:.1f}",
                "count": count,
                "mean_predicted_probability": float(np.mean(p[mask])) if count else None,
                "observed_f1_win_rate": float(np.mean(y[mask])) if count else None,
                "correct_at_0_5_threshold_rate": float(np.mean(((p[mask] >= 0.5).astype(int) == y[mask]))) if count else None,
            }
        )
    return {"bucket_edges": bucket_edges, "buckets": rows}


def confidence_bucket_report(
    y_true: pd.Series,
    probabilities: np.ndarray,
    bucket_edges: list[float] | None = None,
) -> dict[str, Any]:
    if bucket_edges is None:
        bucket_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    confidence = np.maximum(p, 1.0 - p)
    predictions = (p >= 0.5).astype(int)
    correct = predictions == y
    rows = []
    for left, right in zip(bucket_edges[:-1], bucket_edges[1:]):
        if right == bucket_edges[-1]:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        count = int(mask.sum())
        rows.append(
            {
                "confidence_bucket": f"{left:.1f}-{right:.1f}",
                "count": count,
                "mean_confidence": float(np.mean(confidence[mask])) if count else None,
                "accuracy": float(np.mean(correct[mask])) if count else None,
                "mean_predicted_f1_probability": float(np.mean(p[mask])) if count else None,
                "observed_f1_win_rate": float(np.mean(y[mask])) if count else None,
            }
        )
    return {"bucket_edges": bucket_edges, "buckets": rows}


def precision_at_k_report(
    y_true: pd.Series,
    probabilities: np.ndarray,
    ks: list[int] | None = None,
) -> dict[str, Any]:
    if ks is None:
        ks = [25, 50, 100, 250, 500]
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    predictions = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    confidence_order = np.argsort(confidence)[::-1]
    f1_order = np.argsort(p)[::-1]
    rows = []
    f1_rows = []
    for k in ks:
        k_eff = min(k, len(y))
        idx = confidence_order[:k_eff]
        f1_idx = f1_order[:k_eff]
        rows.append(
            {
                "k": int(k_eff),
                "mean_confidence": float(np.mean(confidence[idx])),
                "accuracy": float(np.mean(predictions[idx] == y[idx])),
            }
        )
        f1_rows.append(
            {
                "k": int(k_eff),
                "mean_predicted_f1_probability": float(np.mean(p[f1_idx])),
                "observed_f1_win_rate": float(np.mean(y[f1_idx])),
            }
        )
    return {
        "top_k_by_confidence": rows,
        "top_k_by_f1_probability": f1_rows,
    }


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.005, 20.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "extra_trees":
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 200, step=5),
            "max_depth": trial.suggest_categorical(
                "max_depth",
                [None, 3, 5, 7, 9, 12, 16, 20, 24, 30],
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.25, 0.4, 0.6, 0.8, 1.0],
            ),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced", "balanced_subsample"],
            ),
            "bootstrap": bootstrap,
        }
        if bootstrap:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
        return params
    if model_name == "linear_svm":
        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    raise ValueError(model_name)


def default_params(model_name: str) -> dict[str, Any]:
    defaults = {
        "logistic_regression": {"C": 0.2, "class_weight": None},
        "extra_trees": {
            "n_estimators": 120,
            "max_depth": 8,
            "min_samples_leaf": 18,
            "max_features": "sqrt",
            "class_weight": None,
        },
        "linear_svm": {"C": 0.1, "class_weight": None},
    }
    return defaults[model_name]


def make_base_model(model_name: str, params: dict[str, Any]) -> Pipeline:
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        **params,
                    ),
                ),
            ]
        )
    if model_name == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        max_iter=12000,
                        random_state=RANDOM_STATE,
                        dual="auto",
                        **params,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def fit_prefit_sigmoid_calibrator(
    base_model: Pipeline,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="sigmoid",
            cv=None,
        )
    else:
        calibrator = CalibratedClassifierCV(
            estimator=base_model,
            method="sigmoid",
            cv="prefit",
        )
    calibrator.fit(X_calibration, y_calibration)
    return calibrator


def evaluate_params_on_folds(
    model_name: str,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    diff_features: list[str],
) -> dict[str, Any]:
    fold_results = []
    for fold in folds:
        X_fit, y_fit = symmetric_training_set(
            train_frame.loc[fold["fit_mask"], features],
            train_frame.loc[fold["fit_mask"], "target_f1_win"],
            diff_features,
        )
        X_cal, y_cal = symmetric_training_set(
            train_frame.loc[fold["calibration_mask"], features],
            train_frame.loc[fold["calibration_mask"], "target_f1_win"],
            diff_features,
        )
        X_valid = train_frame.loc[fold["validation_mask"], features]
        y_valid = train_frame.loc[fold["validation_mask"], "target_f1_win"]

        base_model = make_base_model(model_name, params)
        base_model.fit(X_fit, y_fit)
        calibrated = fit_prefit_sigmoid_calibrator(base_model, X_cal, y_cal)
        probabilities = calibrated.predict_proba(X_valid)[:, 1]
        metrics = metric_block(y_valid, probabilities)
        fold_results.append(
            {
                **{key: fold[key] for key in fold if key.endswith("_max") or key.endswith("_min") or key == "fold"},
                "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
                "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
                "validation_rows": int(len(X_valid)),
                "metrics": metrics,
            }
        )

    return {
        "folds": fold_results,
        "mean_log_loss": float(np.mean([fold["metrics"]["log_loss"] for fold in fold_results])),
        "mean_roc_auc": float(np.mean([fold["metrics"]["roc_auc"] for fold in fold_results])),
        "mean_brier_score": float(np.mean([fold["metrics"]["brier_score"] for fold in fold_results])),
        "mean_accuracy": float(np.mean([fold["metrics"]["accuracy"] for fold in fold_results])),
    }


def tune_model(
    model_name: str,
    train_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    diff_features: list[str],
    trials: int,
) -> dict[str, Any]:
    if trials <= 0:
        params = default_params(model_name)
        cv = evaluate_params_on_folds(model_name, params, train_frame, folds, features, diff_features)
        return {"best_params": params, "rolling_cv": cv, "trials": 0}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        cv = evaluate_params_on_folds(model_name, params, train_frame, folds, features, diff_features)
        return cv["mean_log_loss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    cv = evaluate_params_on_folds(
        model_name,
        study.best_params,
        train_frame,
        folds,
        features,
        diff_features,
    )
    return {
        "best_params": study.best_params,
        "best_value_mean_log_loss": float(study.best_value),
        "rolling_cv": cv,
        "trials": trials,
    }


def holdout_evaluation(
    model_name: str,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    features: list[str],
    diff_features: list[str],
) -> dict[str, Any]:
    fit_mask, calibration_mask, calibration_cutoff = temporal_fit_calibration_split(train_frame)
    X_fit, y_fit = symmetric_training_set(
        train_frame.loc[fit_mask, features],
        train_frame.loc[fit_mask, "target_f1_win"],
        diff_features,
    )
    X_cal, y_cal = symmetric_training_set(
        train_frame.loc[calibration_mask, features],
        train_frame.loc[calibration_mask, "target_f1_win"],
        diff_features,
    )
    X_test = test_frame[features]
    y_test = test_frame["target_f1_win"]

    base_model = make_base_model(model_name, params)
    base_model.fit(X_fit, y_fit)
    calibrated = fit_prefit_sigmoid_calibrator(base_model, X_cal, y_cal)
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    metrics = metric_block(y_test, probabilities)
    return {
        "metrics": metrics,
        "calibration_curve": calibration_summary(y_test, probabilities),
        "probability_buckets": probability_bucket_report(y_test, probabilities),
        "confidence_buckets": confidence_bucket_report(y_test, probabilities),
        "precision_at_k": precision_at_k_report(y_test, probabilities),
        "calibration_cutoff_fit_lte": calibration_cutoff.strftime("%Y-%m-%d"),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "base_model": base_model,
        "calibrated_model": calibrated,
    }


def model_importance(base_model: Pipeline, feature_names: list[str]) -> list[dict[str, Any]]:
    classifier = base_model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        key = "coefficient"
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        key = "feature_importance"
    else:
        return []
    return sorted(
        [{"feature": feature, key: float(value)} for feature, value in zip(feature_names, values)],
        key=lambda row: abs(float(row[key])),
        reverse=True,
    )


def train_final_artifact(
    model_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    diff_features: list[str],
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    fit_mask, calibration_mask, calibration_cutoff = temporal_fit_calibration_split(frame)
    X_fit, y_fit = symmetric_training_set(
        frame.loc[fit_mask, features],
        frame.loc[fit_mask, "target_f1_win"],
        diff_features,
    )
    X_cal, y_cal = symmetric_training_set(
        frame.loc[calibration_mask, features],
        frame.loc[calibration_mask, "target_f1_win"],
        diff_features,
    )
    base_model = make_base_model(model_name, params)
    base_model.fit(X_fit, y_fit)
    calibrated = fit_prefit_sigmoid_calibrator(base_model, X_cal, y_cal)
    return calibrated, {
        "artifact_calibration_cutoff_fit_lte": calibration_cutoff.strftime("%Y-%m-%d"),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
    }


def run_experiment(trials: int, folds: int, model_names: list[str]) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame = build_feature_frame(fights)
    features = candidate_features()
    diff_features = [feature for feature in features if feature.endswith("_diff")]
    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    cv_folds = rolling_temporal_folds(train_frame, n_folds=folds)

    model_results = {}
    for model_name in model_names:
        tuning = tune_model(
            model_name,
            train_frame,
            cv_folds,
            features,
            diff_features,
            trials,
        )
        holdout = holdout_evaluation(
            model_name,
            tuning["best_params"],
            train_frame,
            test_frame,
            features,
            diff_features,
        )
        model_artifact_path = ARTIFACT_DIR / f"ufc_candidate_{model_name}_calibrated.joblib"
        final_model, final_training = train_final_artifact(
            model_name,
            tuning["best_params"],
            frame,
            features,
            diff_features,
        )
        artifact = {
            "model": final_model,
            "feature_columns": features,
            "diff_feature_columns": diff_features,
            "feature_medians": frame[features].median(numeric_only=True).to_dict(),
            "metadata": {
                "model_name": model_name,
                "best_params": tuning["best_params"],
                "feature_set": "cumulative_plus_last_fight",
                "feature_count": len(features),
                "data_path": str(DATA_PATH),
                "dataset_rows_with_target": int(len(frame)),
                "dataset_min_date": fights["event_date"].min().strftime("%Y-%m-%d"),
                "dataset_max_date": fights["event_date"].max().strftime("%Y-%m-%d"),
                "calibration_method": "sigmoid",
                **final_training,
            },
        }
        joblib.dump(artifact, model_artifact_path)

        model_results[model_name] = {
            **tuning,
            "holdout": {
                "metrics": holdout["metrics"],
                "calibration_curve": holdout["calibration_curve"],
                "probability_buckets": holdout["probability_buckets"],
                "confidence_buckets": holdout["confidence_buckets"],
                "precision_at_k": holdout["precision_at_k"],
                "calibration_cutoff_fit_lte": holdout["calibration_cutoff_fit_lte"],
                "fit_rows_after_symmetric_augmentation": holdout["fit_rows_after_symmetric_augmentation"],
                "calibration_rows_after_symmetric_augmentation": holdout["calibration_rows_after_symmetric_augmentation"],
                "test_rows": holdout["test_rows"],
            },
            "model_importance": model_importance(holdout["base_model"], features)[:25],
            "artifact_path": str(model_artifact_path),
        }

    best_cv_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["rolling_cv"]["mean_log_loss"],
    )[0]
    best_holdout_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["holdout"]["metrics"]["log_loss"],
    )[0]
    best_cv_artifact = joblib.load(model_results[best_cv_model_name]["artifact_path"])
    best_holdout_artifact = joblib.load(model_results[best_holdout_model_name]["artifact_path"])
    joblib.dump(best_cv_artifact, BEST_CV_MODEL_PATH)
    joblib.dump(best_holdout_artifact, BEST_HOLDOUT_MODEL_PATH)

    ranked = sorted(
        [
            {
                "model": model_name,
                "feature_count": len(features),
                **result["holdout"]["metrics"],
                "rolling_cv_mean_log_loss": result["rolling_cv"]["mean_log_loss"],
                "artifact_path": result["artifact_path"],
            }
            for model_name, result in model_results.items()
        ],
        key=lambda row: row["log_loss"],
    )

    baseline_metrics_path = ARTIFACT_DIR / "ufc_extra_trees_calibrated_metrics.json"
    baseline_metrics = None
    if baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))

    report = {
        "data_path": str(DATA_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "feature_set": "cumulative_plus_last_fight",
        "feature_count": len(features),
        "features": features,
        "models": model_names,
        "optuna_trials_per_model": int(trials),
        "rolling_cv_folds": int(len(cv_folds)),
        "rows_total_with_target": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "test_date_min": test_frame["event_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": test_frame["event_date"].max().strftime("%Y-%m-%d"),
        "leakage_policy": [
            "All candidate features are pre-fight features built from shifted fighter history.",
            "Rolling CV folds train on older dates, calibrate on the next date block, and validate on the following block.",
            "Final holdout uses dates after the 80% chronological train cutoff.",
            "Each base model is sigmoid-calibrated on a temporal calibration block.",
            "Training/calibration rows are symmetrically augmented with swapped fighters.",
        ],
        "model_results": model_results,
        "ranked_by_holdout_log_loss": ranked,
        "best_model_by_rolling_cv_log_loss": best_cv_model_name,
        "best_model_by_holdout_log_loss": best_holdout_model_name,
        "best_cv_model_artifact_path": str(BEST_CV_MODEL_PATH),
        "best_holdout_model_artifact_path": str(BEST_HOLDOUT_MODEL_PATH),
        "reference_15_feature_artifact": baseline_metrics,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate UFC models on cumulative_plus_last_fight features."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=8,
        help="Optuna trials per model. Use 0 for fixed default hyperparameters.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Rolling temporal CV folds used inside the training period.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Subset of candidate models to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(trials=args.trials, folds=args.folds, model_names=args.models)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
