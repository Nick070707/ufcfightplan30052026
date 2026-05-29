"""UFC final pipeline (для текущей серии экспериментов).

Конфигурация:

  - 92 признака: всё из ufc_matchup_pipeline (93) минус `southpaw_reach_diff`.
    Stance-сигнал остаётся только как чистый XOR-индикатор `stance_mismatch`,
    без множителя на reach. Reach остаётся отдельно как `reach_cm_diff`.
  - Две модели: LogisticRegression и HistGradientBoosting.
  - Sigmoid calibration (возврат к тому, что работало в rated).
  - Optuna tuning + rolling temporal CV + symmetric augmentation.

Запуск:
    python ufc_final_pipeline.py --trials 30 --folds 4
"""

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ufc_ablation_analysis import (
    ARTIFACT_DIR,
    DATA_PATH,
    load_clean_fights,
    symmetric_training_set,
)
from ufc_advanced_pipeline import RANDOM_STATE
from ufc_matchup_pipeline import (
    ANTISYM_MATCHUP_FEATURES,
    SYM_MATCHUP_FEATURES,
    build_matchup_frame,
)
from train_candidate_models import (
    calibration_summary,
    chronological_split,
    confidence_bucket_report,
    fit_prefit_sigmoid_calibrator,
    metric_block,
    model_importance,
    precision_at_k_report,
    probability_bucket_report,
    rolling_temporal_folds,
    temporal_fit_calibration_split,
)


RESULTS_PATH = ARTIFACT_DIR / "ufc_final_pipeline_results.json"
BEST_CV_MODEL_PATH = ARTIFACT_DIR / "ufc_final_best_cv_calibrated.joblib"
BEST_HOLDOUT_MODEL_PATH = ARTIFACT_DIR / "ufc_final_best_holdout_calibrated.joblib"

MODEL_NAMES = ["logistic_regression", "hist_gradient_boosting"]

# Фичи, которые удаляем из матчап-набора в финальном пайплайне.
DROPPED_FEATURES: set[str] = {"southpaw_reach_diff"}


def build_final_frame(
    fights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame, features, diff_features = build_matchup_frame(fights)
    features = [f for f in features if f not in DROPPED_FEATURES]
    diff_features = [f for f in diff_features if f not in DROPPED_FEATURES]
    return frame, features, diff_features


# ---------------------------------------------------------------------------
# Model factory: LR + HistGradientBoosting
# ---------------------------------------------------------------------------


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
    if model_name == "hist_gradient_boosting":
        # HGB обрабатывает NaN нативно — imputer не нужен.
        return Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                        **params,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 20.0, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.002, 0.5, log=True
            ),
            "max_iter": trial.suggest_int("max_iter", 50, 2000, step=20),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 127),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 3, 200
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.0, 10.0
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 3, 4, 6, 8, 12, 16]
            ),
        }
    raise ValueError(model_name)


def default_params(model_name: str) -> dict[str, Any]:
    defaults = {
        "logistic_regression": {"C": 0.005, "class_weight": None},
        "hist_gradient_boosting": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "max_depth": None,
        },
    }
    return defaults[model_name]


# ---------------------------------------------------------------------------
# Sigmoid-based training loop
# ---------------------------------------------------------------------------


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

        base = make_base_model(model_name, params)
        base.fit(X_fit, y_fit)
        calibrated = fit_prefit_sigmoid_calibrator(base, X_cal, y_cal)
        probabilities = calibrated.predict_proba(X_valid)[:, 1]
        metrics = metric_block(y_valid, probabilities)
        fold_results.append(
            {
                **{
                    k: fold[k]
                    for k in fold
                    if k.endswith("_max") or k.endswith("_min") or k == "fold"
                },
                "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
                "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
                "validation_rows": int(len(X_valid)),
                "metrics": metrics,
            }
        )
    return {
        "folds": fold_results,
        "mean_log_loss": float(
            np.mean([f["metrics"]["log_loss"] for f in fold_results])
        ),
        "mean_roc_auc": float(
            np.mean([f["metrics"]["roc_auc"] for f in fold_results])
        ),
        "mean_brier_score": float(
            np.mean([f["metrics"]["brier_score"] for f in fold_results])
        ),
        "mean_accuracy": float(
            np.mean([f["metrics"]["accuracy"] for f in fold_results])
        ),
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
        cv = evaluate_params_on_folds(
            model_name, params, train_frame, folds, features, diff_features
        )
        return {"best_params": params, "rolling_cv": cv, "trials": 0}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        cv = evaluate_params_on_folds(
            model_name, params, train_frame, folds, features, diff_features
        )
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
    fit_mask, calibration_mask, cutoff = temporal_fit_calibration_split(
        train_frame
    )
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

    base = make_base_model(model_name, params)
    base.fit(X_fit, y_fit)
    calibrated = fit_prefit_sigmoid_calibrator(base, X_cal, y_cal)
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    metrics = metric_block(y_test, probabilities)
    return {
        "metrics": metrics,
        "calibration_curve": calibration_summary(y_test, probabilities),
        "probability_buckets": probability_bucket_report(y_test, probabilities),
        "confidence_buckets": confidence_bucket_report(y_test, probabilities),
        "precision_at_k": precision_at_k_report(y_test, probabilities),
        "calibration_cutoff_fit_lte": cutoff.strftime("%Y-%m-%d"),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "base_model": base,
        "calibrated_model": calibrated,
    }


def train_final_artifact(
    model_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    diff_features: list[str],
):
    fit_mask, calibration_mask, cutoff = temporal_fit_calibration_split(frame)
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
    base = make_base_model(model_name, params)
    base.fit(X_fit, y_fit)
    calibrated = fit_prefit_sigmoid_calibrator(base, X_cal, y_cal)
    return calibrated, {
        "artifact_calibration_cutoff_fit_lte": cutoff.strftime("%Y-%m-%d"),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
    }


def train_full_data_artifact(
    model_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    diff_features: list[str],
    n_folds: int = 5,
):
    """Финальный артефакт, обучающий базовую модель на МАКСИМАЛЬНОМ объёме
    данных: ни один блок не откладывается.

    В отличие от `train_final_artifact` (база обучается только на fit-части
    ≤ cutoff, хвост идёт лишь на калибровку), здесь используется
    `CalibratedClassifierCV(cv=..., ensemble=False)`:

      - базовая модель (деревья) обучается на **100% боёв** (включая самые
        свежие) — это и есть «максимальный объём»;
      - 2-параметрический sigmoid калибруется на **out-of-fold** предсказаниях
        (`cross_val_predict`), поэтому калибровка остаётся валидной (без
        in-sample оптимизма);
      - итоговый scorer — единая модель (база на всех данных) + один калибратор.

    Фолды group-aware по индексу боя: оригинал и его симметричное зеркало
    (swap бойцов из symmetric_training_set) всегда в одном фолде, иначе зеркало
    «протекало» бы между train и calibration внутри cross_val_predict.
    """
    X = frame[features].reset_index(drop=True)
    y = frame["target_f1_win"].reset_index(drop=True)
    n_fights = len(X)

    X_aug, y_aug = symmetric_training_set(X, y, diff_features)
    # group = индекс боя; symmetric_training_set кладёт зеркало в строку i+n_fights
    groups = np.concatenate([np.arange(n_fights), np.arange(n_fights)])

    splits = list(GroupKFold(n_splits=n_folds).split(X_aug, y_aug, groups))

    calibrated = CalibratedClassifierCV(
        estimator=make_base_model(model_name, params),
        method="sigmoid",
        cv=splits,
        ensemble=False,  # база обучается на ВСЕХ данных; cv — только для OOF-калибровки
    )
    calibrated.fit(X_aug, y_aug)
    return calibrated, {
        "training_scheme": (
            f"base on 100% of fights; sigmoid calibrated on out-of-fold preds "
            f"(GroupKFold k={n_folds}, ensemble=False)"
        ),
        "fights_used_for_base_training": int(n_fights),
        "rows_after_symmetric_augmentation": int(len(X_aug)),
        "n_calibration_folds": int(n_folds),
        "external_holdout": False,
    }


# ---------------------------------------------------------------------------
# Главный прогон
# ---------------------------------------------------------------------------


def _load_reference(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(
    trials: int, folds: int, model_names: list[str]
) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame, features, diff_features = build_final_frame(fights)

    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    cv_folds = rolling_temporal_folds(train_frame, n_folds=folds)

    matchup_features = set(ANTISYM_MATCHUP_FEATURES + SYM_MATCHUP_FEATURES) - DROPPED_FEATURES

    model_results: dict[str, Any] = {}
    for model_name in model_names:
        tuning = tune_model(
            model_name, train_frame, cv_folds, features, diff_features, trials
        )
        holdout = holdout_evaluation(
            model_name,
            tuning["best_params"],
            train_frame,
            test_frame,
            features,
            diff_features,
        )
        artifact_path = (
            ARTIFACT_DIR / f"ufc_final_{model_name}_sigmoid_calibrated.joblib"
        )
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
                "feature_set": (
                    "cumulative_plus_last_fight + position_off_def + ratings + matchup (minus southpaw_reach_diff)"
                ),
                "feature_count": len(features),
                "dropped_features_from_matchup": sorted(DROPPED_FEATURES),
                "matchup_features_used": sorted(matchup_features),
                "data_path": str(DATA_PATH),
                "dataset_rows_with_target": int(len(frame)),
                "dataset_min_date": fights["event_date"]
                .min()
                .strftime("%Y-%m-%d"),
                "dataset_max_date": fights["event_date"]
                .max()
                .strftime("%Y-%m-%d"),
                "calibration_method": "sigmoid",
                **final_training,
            },
        }
        joblib.dump(artifact, artifact_path)

        importances = model_importance(holdout["base_model"], features)
        # Для HGB built-in importance отсутствует (требует permutation), оставим
        # пустой список, если importances не извлеклись.
        model_results[model_name] = {
            **tuning,
            "holdout": {
                "metrics": holdout["metrics"],
                "calibration_curve": holdout["calibration_curve"],
                "probability_buckets": holdout["probability_buckets"],
                "confidence_buckets": holdout["confidence_buckets"],
                "precision_at_k": holdout["precision_at_k"],
                "calibration_cutoff_fit_lte": holdout["calibration_cutoff_fit_lte"],
                "fit_rows_after_symmetric_augmentation": holdout[
                    "fit_rows_after_symmetric_augmentation"
                ],
                "calibration_rows_after_symmetric_augmentation": holdout[
                    "calibration_rows_after_symmetric_augmentation"
                ],
                "test_rows": holdout["test_rows"],
            },
            "model_importance": importances[:40] if importances else [],
            "artifact_path": str(artifact_path),
        }

    best_cv_name = min(
        model_results.items(),
        key=lambda item: item[1]["rolling_cv"]["mean_log_loss"],
    )[0]
    best_holdout_name = min(
        model_results.items(),
        key=lambda item: item[1]["holdout"]["metrics"]["log_loss"],
    )[0]
    joblib.dump(
        joblib.load(model_results[best_cv_name]["artifact_path"]),
        BEST_CV_MODEL_PATH,
    )
    joblib.dump(
        joblib.load(model_results[best_holdout_name]["artifact_path"]),
        BEST_HOLDOUT_MODEL_PATH,
    )

    ranked = sorted(
        [
            {
                "model": model_name,
                "feature_count": len(features),
                **result["holdout"]["metrics"],
                "rolling_cv_mean_log_loss": result["rolling_cv"]["mean_log_loss"],
                "rolling_cv_mean_roc_auc": result["rolling_cv"]["mean_roc_auc"],
                "artifact_path": result["artifact_path"],
            }
            for model_name, result in model_results.items()
        ],
        key=lambda row: row["log_loss"],
    )

    baseline_metrics = _load_reference(
        ARTIFACT_DIR / "ufc_extra_trees_calibrated_metrics.json"
    )
    candidate_results = _load_reference(
        ARTIFACT_DIR / "ufc_candidate_models_results.json"
    )
    rated_results = _load_reference(
        ARTIFACT_DIR / "ufc_rated_pipeline_results.json"
    )
    matchup_lr_results = _load_reference(
        ARTIFACT_DIR / "ufc_matchup_pipeline_results.json"
    )
    matchup_et_results = _load_reference(
        ARTIFACT_DIR / "ufc_matchup_et_pipeline_results.json"
    )

    report = {
        "data_path": str(DATA_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "feature_set": (
            "cumulative_plus_last_fight + position_off_def + ratings + matchup (minus southpaw_reach_diff)"
        ),
        "feature_count": len(features),
        "dropped_features_from_matchup": sorted(DROPPED_FEATURES),
        "matchup_features_used": sorted(matchup_features),
        "features": features,
        "diff_features": diff_features,
        "models": model_names,
        "calibration_method": "sigmoid",
        "optuna_trials_per_model": int(trials),
        "rolling_cv_folds": int(len(cv_folds)),
        "rows_total_with_target": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "test_date_min": test_frame["event_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": test_frame["event_date"].max().strftime("%Y-%m-%d"),
        "leakage_policy": [
            "Все cumulative/position/rating/matchup-агрегаты считаются строго из shifted history.",
            "Rolling CV: train -> calibration -> validation последовательно по датам.",
            "Holdout: даты после 80%-го хронологического сплита.",
            "Sigmoid calibration на отдельном temporal calibration-блоке.",
            "Train + calibration строки симметрично дополняются перестановкой бойцов.",
            "HGB обрабатывает NaN нативно, LR использует median imputation.",
        ],
        "model_results": model_results,
        "ranked_by_holdout_log_loss": ranked,
        "best_model_by_rolling_cv_log_loss": best_cv_name,
        "best_model_by_holdout_log_loss": best_holdout_name,
        "best_cv_model_artifact_path": str(BEST_CV_MODEL_PATH),
        "best_holdout_model_artifact_path": str(BEST_HOLDOUT_MODEL_PATH),
        "reference_baseline_15_feature": baseline_metrics,
        "reference_candidate_pipeline_ranked": (
            candidate_results.get("ranked_by_holdout_log_loss")
            if candidate_results
            else None
        ),
        "reference_rated_pipeline_ranked": (
            rated_results.get("ranked_by_holdout_log_loss")
            if rated_results
            else None
        ),
        "reference_matchup_lr_holdout": (
            matchup_lr_results.get("holdout", {}).get("metrics")
            if matchup_lr_results
            else None
        ),
        "reference_matchup_et_holdout": (
            matchup_et_results.get("holdout", {}).get("metrics")
            if matchup_et_results
            else None
        ),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final pipeline: 92 features (matchup minus southpaw_reach_diff), "
            "LogReg + HistGradientBoosting, sigmoid calibration."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Optuna trials per model.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Rolling temporal CV folds inside the training period.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_NAMES,
        choices=MODEL_NAMES,
        help="Subset of models to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(
        trials=args.trials, folds=args.folds, model_names=args.models
    )
    summary = {
        "feature_count": report["feature_count"],
        "rolling_cv_folds": report["rolling_cv_folds"],
        "best_model_by_rolling_cv_log_loss": report[
            "best_model_by_rolling_cv_log_loss"
        ],
        "best_model_by_holdout_log_loss": report[
            "best_model_by_holdout_log_loss"
        ],
        "ranked_by_holdout_log_loss": report["ranked_by_holdout_log_loss"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
