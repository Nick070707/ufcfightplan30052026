"""UFC matchup pipeline.

Шаг поверх ufc_rated_pipeline.py:

  - все 82 признака из rated-пайплайна (cumulative_plus_last_fight +
    position offense/defense + opponent-adjusted ratings);
  - новые matchup-признаки:
      * style: style_lean per fighter, style_lean_diff, style_clash;
      * stance: stance_mismatch (южн./ортодокс. матчап), southpaw_reach_diff;
      * weight cuts: weight_cut_diff, weight_cut_max, weight_cut_sum,
        weight_class_lbs;
      * layoff: layoff_max / layoff_min / layoff_abs_diff (оба бойца одновременно);
  - isotonic calibration вместо sigmoid;
  - для скорости тюним только LogisticRegression.

Запуск:
    python ufc_matchup_pipeline.py --trials 40 --folds 4
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

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None

from ufc_ablation_analysis import (
    ARTIFACT_DIR,
    DATA_PATH,
    load_clean_fights,
    symmetric_training_set,
    unique_features,
)
from ufc_advanced_pipeline import RANDOM_STATE, numeric
from ufc_rated_pipeline import (
    BASE_GROUPS,
    POSITIONS,
    POSITION_DEFENSE_METRICS,
    POSITION_OFFENSE_METRICS,
    RATING_FEATURE_NAMES,
    build_rated_frame,
)
from train_candidate_models import (
    calibration_summary,
    chronological_split,
    confidence_bucket_report,
    make_base_model,
    metric_block,
    model_importance,
    precision_at_k_report,
    probability_bucket_report,
    rolling_temporal_folds,
    temporal_fit_calibration_split,
)


RESULTS_PATH = ARTIFACT_DIR / "ufc_matchup_pipeline_results.json"
ARTIFACT_PATH = ARTIFACT_DIR / "ufc_matchup_logistic_regression_isotonic_calibrated.joblib"


# ---------------------------------------------------------------------------
# Matchup feature engineering
# ---------------------------------------------------------------------------

# Порядок важен: "light heavyweight" должен матчиться раньше "heavyweight".
WEIGHT_CLASS_LBS: list[tuple[str, float]] = [
    ("light heavyweight", 205.0),
    ("strawweight", 115.0),
    ("flyweight", 125.0),
    ("bantamweight", 135.0),
    ("featherweight", 145.0),
    ("lightweight", 155.0),
    ("welterweight", 170.0),
    ("middleweight", 185.0),
    ("heavyweight", 265.0),
]


def weight_class_to_lbs(value: Any) -> float:
    if not isinstance(value, str):
        return np.nan
    low = value.lower()
    for name, lbs in WEIGHT_CLASS_LBS:
        if name in low:
            return lbs
    return np.nan


def compute_per_fighter_layoff(fights: pd.DataFrame) -> pd.DataFrame:
    """Per-fighter days_since_last_fight (shifted = pre-fight)."""
    frames = []
    for side in (1, 2):
        frames.append(
            pd.DataFrame(
                {
                    "event_date": fights["event_date"],
                    "fight_url": fights["fight_url"],
                    "fighter_id": fights[f"f_{side}_id"],
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    event_level = (
        long.groupby(["fighter_id", "event_date"], as_index=False)
        .agg(fight_count=("fight_url", "count"))
        .sort_values(["fighter_id", "event_date"])
    )
    grouped = event_level.groupby("fighter_id", group_keys=False)
    previous = grouped["event_date"].shift(1)
    event_level["days_since_last_fight"] = (
        event_level["event_date"] - previous
    ).dt.days
    return event_level[["fighter_id", "event_date", "days_since_last_fight"]]


ANTISYM_MATCHUP_FEATURES = [
    "style_lean_diff",
    "southpaw_reach_diff",
    "weight_cut_diff",
]

SYM_MATCHUP_FEATURES = [
    "style_clash",
    "stance_mismatch",
    "layoff_max",
    "layoff_min",
    "layoff_abs_diff",
    "weight_cut_max",
    "weight_cut_sum",
    "weight_class_lbs",
]


def add_matchup_features(
    frame: pd.DataFrame, fights: pd.DataFrame, layoffs: pd.DataFrame
) -> pd.DataFrame:
    """Расширяет rated frame matchup-признаками."""
    for side in (1, 2):
        renamed = layoffs.rename(
            columns={"days_since_last_fight": f"f_{side}_layoff_days"}
        )
        frame = frame.merge(
            renamed,
            how="left",
            left_on=[f"f_{side}_id", "event_date"],
            right_on=["fighter_id", "event_date"],
        ).drop(columns=["fighter_id"])

    extra_cols = [
        "f_1_fighter_stance",
        "f_2_fighter_stance",
        "f_1_fighter_weight_lbs",
        "f_2_fighter_weight_lbs",
        "f_1_fighter_reach_cm",
        "f_2_fighter_reach_cm",
        "weight_class",
    ]
    frame = frame.merge(
        fights[["fight_url", *extra_cols]], on="fight_url", how="left"
    )

    for side in (1, 2):
        stance = (
            frame[f"f_{side}_fighter_stance"].fillna("").astype(str).str.lower()
        )
        frame[f"f_{side}_southpaw"] = stance.eq("southpaw").astype(int)

    frame["weight_class_lbs"] = frame["weight_class"].map(weight_class_to_lbs)

    for side in (1, 2):
        cut = numeric(frame[f"f_{side}_fighter_weight_lbs"]) - frame[
            "weight_class_lbs"
        ]
        frame[f"f_{side}_weight_cut_lbs"] = cut.clip(lower=0)

    for side in (1, 2):
        frame[f"f_{side}_style_lean"] = (
            frame[f"f_{side}_pre_striking_elo"]
            - frame[f"f_{side}_pre_grappling_elo"]
        ) / 100.0

    # --- Антисимметричные diff'ы (знак инвертируется при swap f_1<->f_2) ---
    frame["style_lean_diff"] = (
        frame["f_1_style_lean"] - frame["f_2_style_lean"]
    )
    frame["southpaw_reach_diff"] = (
        frame["f_1_southpaw"] * numeric(frame["f_1_fighter_reach_cm"])
        - frame["f_2_southpaw"] * numeric(frame["f_2_fighter_reach_cm"])
    )
    frame["weight_cut_diff"] = (
        frame["f_1_weight_cut_lbs"] - frame["f_2_weight_cut_lbs"]
    )

    # --- Симметричные matchup-признаки (не меняются при swap f_1<->f_2) ---
    frame["style_clash"] = (
        frame["f_1_style_lean"] * frame["f_2_style_lean"]
    )
    frame["stance_mismatch"] = (
        frame["f_1_southpaw"] != frame["f_2_southpaw"]
    ).astype(int)
    f1d = frame["f_1_layoff_days"]
    f2d = frame["f_2_layoff_days"]
    frame["layoff_max"] = pd.concat([f1d, f2d], axis=1).max(axis=1)
    frame["layoff_min"] = pd.concat([f1d, f2d], axis=1).min(axis=1)
    frame["layoff_abs_diff"] = (f1d - f2d).abs()
    frame["weight_cut_max"] = pd.concat(
        [frame["f_1_weight_cut_lbs"], frame["f_2_weight_cut_lbs"]], axis=1
    ).max(axis=1)
    frame["weight_cut_sum"] = (
        frame["f_1_weight_cut_lbs"] + frame["f_2_weight_cut_lbs"]
    )
    return frame


def build_matchup_frame(
    fights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame, rated_features, _ = build_rated_frame(fights)
    layoffs = compute_per_fighter_layoff(fights)
    frame = add_matchup_features(frame, fights, layoffs)
    features = (
        list(rated_features) + ANTISYM_MATCHUP_FEATURES + SYM_MATCHUP_FEATURES
    )
    diff_features = [feat for feat in features if feat.endswith("_diff")]
    return frame, features, diff_features


# ---------------------------------------------------------------------------
# Isotonic-based training loop
# ---------------------------------------------------------------------------


def fit_prefit_isotonic_calibrator(
    base_model, X_calibration: pd.DataFrame, y_calibration: pd.Series
) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="isotonic",
            cv=None,
        )
    else:
        cal = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv="prefit",
        )
    cal.fit(X_calibration, y_calibration)
    return cal


def evaluate_params_on_folds_isotonic(
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
        calibrated = fit_prefit_isotonic_calibrator(base, X_cal, y_cal)
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


def suggest_lr_params(trial: optuna.Trial) -> dict[str, Any]:
    # Расширенный диапазон: предыдущая модель садилась на C=0.005 -
    # даём Optuna уйти ещё сильнее в L2.
    return {
        "C": trial.suggest_float("C", 1e-4, 20.0, log=True),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        ),
    }


def tune_logistic_regression(
    train_frame: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    diff_features: list[str],
    trials: int,
) -> dict[str, Any]:
    if trials <= 0:
        params = {"C": 0.005, "class_weight": None}
        cv = evaluate_params_on_folds_isotonic(
            "logistic_regression",
            params,
            train_frame,
            folds,
            features,
            diff_features,
        )
        return {"best_params": params, "rolling_cv": cv, "trials": 0}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lr_params(trial)
        cv = evaluate_params_on_folds_isotonic(
            "logistic_regression",
            params,
            train_frame,
            folds,
            features,
            diff_features,
        )
        return cv["mean_log_loss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    cv = evaluate_params_on_folds_isotonic(
        "logistic_regression",
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


def holdout_evaluation_isotonic(
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

    base = make_base_model("logistic_regression", params)
    base.fit(X_fit, y_fit)
    calibrated = fit_prefit_isotonic_calibrator(base, X_cal, y_cal)
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


def train_final_artifact_isotonic(
    params: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    diff_features: list[str],
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
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
    base = make_base_model("logistic_regression", params)
    base.fit(X_fit, y_fit)
    calibrated = fit_prefit_isotonic_calibrator(base, X_cal, y_cal)
    return calibrated, {
        "artifact_calibration_cutoff_fit_lte": cutoff.strftime("%Y-%m-%d"),
        "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
        "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
    }


# ---------------------------------------------------------------------------
# Главный прогон
# ---------------------------------------------------------------------------


def _load_reference(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_experiment(trials: int, folds: int) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame, features, diff_features = build_matchup_frame(fights)
    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    cv_folds = rolling_temporal_folds(train_frame, n_folds=folds)

    tuning = tune_logistic_regression(
        train_frame, cv_folds, features, diff_features, trials
    )
    holdout = holdout_evaluation_isotonic(
        tuning["best_params"], train_frame, test_frame, features, diff_features
    )
    final_model, final_training = train_final_artifact_isotonic(
        tuning["best_params"], frame, features, diff_features
    )

    artifact = {
        "model": final_model,
        "feature_columns": features,
        "diff_feature_columns": diff_features,
        "feature_medians": frame[features].median(numeric_only=True).to_dict(),
        "metadata": {
            "model_name": "logistic_regression",
            "best_params": tuning["best_params"],
            "feature_set": (
                "cumulative_plus_last_fight + position_off_def + ratings + matchup"
            ),
            "feature_count": len(features),
            "rated_features": len(features)
            - len(ANTISYM_MATCHUP_FEATURES)
            - len(SYM_MATCHUP_FEATURES),
            "matchup_diff_features": ANTISYM_MATCHUP_FEATURES,
            "matchup_sym_features": SYM_MATCHUP_FEATURES,
            "data_path": str(DATA_PATH),
            "dataset_rows_with_target": int(len(frame)),
            "dataset_min_date": fights["event_date"]
            .min()
            .strftime("%Y-%m-%d"),
            "dataset_max_date": fights["event_date"]
            .max()
            .strftime("%Y-%m-%d"),
            "calibration_method": "isotonic",
            **final_training,
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    baseline_metrics = _load_reference(
        ARTIFACT_DIR / "ufc_extra_trees_calibrated_metrics.json"
    )
    candidate_results = _load_reference(
        ARTIFACT_DIR / "ufc_candidate_models_results.json"
    )
    rated_results = _load_reference(
        ARTIFACT_DIR / "ufc_rated_pipeline_results.json"
    )

    report = {
        "data_path": str(DATA_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "feature_set": (
            "cumulative_plus_last_fight + position_off_def + ratings + matchup"
        ),
        "feature_count": len(features),
        "features": features,
        "diff_features": diff_features,
        "antisymmetric_matchup_features": ANTISYM_MATCHUP_FEATURES,
        "symmetric_matchup_features": SYM_MATCHUP_FEATURES,
        "model": "logistic_regression",
        "calibration_method": "isotonic",
        "optuna_trials": int(trials),
        "rolling_cv_folds": int(len(cv_folds)),
        "rows_total_with_target": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "test_date_min": test_frame["event_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": test_frame["event_date"].max().strftime("%Y-%m-%d"),
        "leakage_policy": [
            "Все cumulative/position/rating-агрегаты считаются строго из shifted history.",
            "Matchup-признаки строятся из тех же pre-fight значений, без подсматривания вперёд.",
            "Rolling CV: train -> calibration -> validation последовательно по датам.",
            "Holdout: даты после 80%-го хронологического сплита.",
            "Isotonic calibration на отдельном temporal calibration-блоке.",
            "Train + calibration строки симметрично дополняются перестановкой бойцов.",
        ],
        "tuning": {
            "best_params": tuning["best_params"],
            "best_value_mean_log_loss": tuning.get(
                "best_value_mean_log_loss"
            ),
            "rolling_cv": tuning["rolling_cv"],
        },
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
        "model_importance": model_importance(holdout["base_model"], features)[
            :40
        ],
        "artifact_path": str(ARTIFACT_PATH),
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
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train UFC LogisticRegression with rated features + matchup "
            "features + isotonic calibration."
        )
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Optuna trials for LogisticRegression.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Rolling temporal CV folds used inside the training period.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(trials=args.trials, folds=args.folds)
    summary = {
        "feature_count": report["feature_count"],
        "rolling_cv_folds": report["rolling_cv_folds"],
        "best_params": report["tuning"]["best_params"],
        "rolling_cv_mean_log_loss": report["tuning"]["rolling_cv"][
            "mean_log_loss"
        ],
        "rolling_cv_mean_roc_auc": report["tuning"]["rolling_cv"][
            "mean_roc_auc"
        ],
        "holdout_metrics": report["holdout"]["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
