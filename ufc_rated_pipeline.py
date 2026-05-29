"""UFC rated pipeline.

Расширение лучшего candidate-сетапа из train_candidate_models.py:

  - feature set cumulative_plus_last_fight (43 признака) - база;
  - per-position cumulative offense + defense (head / body / leg / distance /
    clinch / ground) - 30 новых diff-признаков;
  - opponent-adjusted ratings: overall Elo, Glicko-1 (rating + RD),
    skill Elo (striking / grappling / KO-power-vs-resistance) и
    strength-of-schedule - ~9 новых diff-признаков.

Сохраняется:
  - rolling temporal CV + Optuna + sigmoid calibration;
  - temporal holdout с теми же датами, что в финальном эксперименте;
  - symmetric augmentation тренировочных и калибровочных строк;
  - leak-policy: все рейтинги и cumulative-агрегаты считаются строго
    из прошлого, по shifted history.

Запуск:
    python ufc_rated_pipeline.py --trials 25 --folds 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import optuna
import pandas as pd

from ufc_ablation_analysis import (
    ARTIFACT_DIR,
    DATA_PATH,
    build_feature_frame,
    load_clean_fights,
    symmetric_training_set,
    unique_features,
)
from ufc_advanced_pipeline import (
    POSITION_STATS,
    RANDOM_STATE,
    numeric,
    round_stat_sum,
    safe_rate,
)
from train_candidate_models import (
    calibration_summary,
    chronological_split,
    confidence_bucket_report,
    evaluate_params_on_folds,
    fit_prefit_sigmoid_calibrator,
    holdout_evaluation,
    make_base_model,
    metric_block,
    model_importance,
    precision_at_k_report,
    probability_bucket_report,
    rolling_temporal_folds,
    temporal_fit_calibration_split,
    train_final_artifact,
)


RESULTS_PATH = ARTIFACT_DIR / "ufc_rated_pipeline_results.json"
BEST_CV_MODEL_PATH = ARTIFACT_DIR / "ufc_rated_best_cv_calibrated.joblib"
BEST_HOLDOUT_MODEL_PATH = ARTIFACT_DIR / "ufc_rated_best_holdout_calibrated.joblib"

MODEL_NAMES = ["logistic_regression", "extra_trees", "linear_svm"]

BASE_GROUPS = [
    "physical",
    "context",
    "cumulative_results",
    "cumulative_style",
    "last_fight_state",
]

POSITIONS = ["head", "body", "leg", "distance", "clinch", "ground"]

POSITION_OFFENSE_METRICS = [
    "landed_per_min",
    "attempted_per_min",
    "accuracy",
]
POSITION_DEFENSE_METRICS = [
    "absorbed_per_min",
    "defense",
]

RATING_FEATURE_NAMES = [
    "pre_elo",
    "pre_glicko_r",
    "pre_glicko_rd",
    "pre_striking_elo",
    "pre_grappling_elo",
    "pre_ko_elo",
    "pre_sos_avg_opp_elo",
    "pre_n_fights",
    "pre_glicko_minus_elo",
]


# ---------------------------------------------------------------------------
# Ratings engine
# ---------------------------------------------------------------------------

INIT_ELO = 1500.0
K_OVERALL = 24.0
K_SKILL = 16.0
INIT_GLICKO_R = 1500.0
INIT_GLICKO_RD = 350.0
GLICKO_RD_FLOOR = 30.0
GLICKO_Q = math.log(10) / 400.0


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result):
        return 0.0
    return result


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_update_pair(
    rating_a: float, rating_b: float, score_a: float, k: float
) -> tuple[float, float]:
    expected_a = elo_expected(rating_a, rating_b)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


def glicko_g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * (GLICKO_Q ** 2) * (rd ** 2) / (math.pi ** 2))


def glicko_expected(rating: float, rating_opp: float, rd_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-glicko_g(rd_opp) * (rating - rating_opp) / 400.0))


def glicko_update(
    rating: float,
    rd: float,
    rating_opp: float,
    rd_opp: float,
    score: float,
) -> tuple[float, float]:
    g = glicko_g(rd_opp)
    expected = glicko_expected(rating, rating_opp, rd_opp)
    expected = min(max(expected, 1e-6), 1 - 1e-6)
    d2 = 1.0 / (GLICKO_Q ** 2 * g ** 2 * expected * (1.0 - expected))
    rd_sq_inv = 1.0 / (rd ** 2)
    new_rating = rating + (GLICKO_Q / (rd_sq_inv + 1.0 / d2)) * g * (score - expected)
    new_rd = math.sqrt(1.0 / (rd_sq_inv + 1.0 / d2))
    new_rd = max(min(new_rd, INIT_GLICKO_RD), GLICKO_RD_FLOOR)
    return new_rating, new_rd


def grappling_score_row(row: pd.Series, side: int) -> float:
    """Композит грэпплинга: тейкдауны + сабмишн-попытки + контроль (минуты)."""
    return (
        _safe_float(row.get(f"f_{side}_takedown_succ")) * 3.0
        + _safe_float(row.get(f"f_{side}_submission_att")) * 2.0
        + _safe_float(row.get(f"f_{side}_ctrl_time_sec")) / 60.0
    )


def result_is_ko_tko(result_value: Any) -> bool:
    if not isinstance(result_value, str):
        return False
    lowered = result_value.lower()
    return ("ko" in lowered) or ("tko" in lowered) or ("doctor" in lowered)


def _new_fighter_state() -> dict[str, float]:
    return {
        "elo": INIT_ELO,
        "g_r": INIT_GLICKO_R,
        "g_rd": INIT_GLICKO_RD,
        "str_elo": INIT_ELO,
        "grp_elo": INIT_ELO,
        "ko_elo": INIT_ELO,
        "n": 0.0,
        "sos_sum": 0.0,
    }


def compute_pre_fight_ratings(fights: pd.DataFrame) -> pd.DataFrame:
    """Прокручивает бои в хронологическом порядке и собирает pre-fight рейтинги.

    Возвращает DataFrame с одной строкой на бой и колонками
    f_{1/2}_<RATING_FEATURE_NAMES>. Все значения - state перед боем.
    Обновления рейтингов делаются только после фиксации pre-fight.
    """
    sorted_fights = fights.sort_values(
        ["event_date", "event_name", "fight_url"]
    ).reset_index(drop=True)

    state: dict[str, dict[str, float]] = defaultdict(_new_fighter_state)
    rows: list[dict[str, Any]] = []

    for _, row in sorted_fights.iterrows():
        a_id = row["f_1_id"]
        b_id = row["f_2_id"]
        sa = state[a_id]
        sb = state[b_id]

        pre = {
            "fight_url": row["fight_url"],
            "f_1_pre_elo": sa["elo"],
            "f_2_pre_elo": sb["elo"],
            "f_1_pre_glicko_r": sa["g_r"],
            "f_2_pre_glicko_r": sb["g_r"],
            "f_1_pre_glicko_rd": sa["g_rd"],
            "f_2_pre_glicko_rd": sb["g_rd"],
            "f_1_pre_striking_elo": sa["str_elo"],
            "f_2_pre_striking_elo": sb["str_elo"],
            "f_1_pre_grappling_elo": sa["grp_elo"],
            "f_2_pre_grappling_elo": sb["grp_elo"],
            "f_1_pre_ko_elo": sa["ko_elo"],
            "f_2_pre_ko_elo": sb["ko_elo"],
            "f_1_pre_sos_avg_opp_elo": (sa["sos_sum"] / sa["n"]) if sa["n"] > 0 else np.nan,
            "f_2_pre_sos_avg_opp_elo": (sb["sos_sum"] / sb["n"]) if sb["n"] > 0 else np.nan,
            "f_1_pre_n_fights": float(sa["n"]),
            "f_2_pre_n_fights": float(sb["n"]),
            "f_1_pre_glicko_minus_elo": sa["g_r"] - sa["elo"],
            "f_2_pre_glicko_minus_elo": sb["g_r"] - sb["elo"],
        }
        rows.append(pre)

        score_a = float(row["target_f1_win"])

        # SOS обновляется ДО Elo, чтобы накопить именно pre-fight рейтинг соперника.
        sa["sos_sum"] += sb["elo"]
        sb["sos_sum"] += sa["elo"]
        sa["n"] += 1.0
        sb["n"] += 1.0

        # Overall Elo
        sa["elo"], sb["elo"] = elo_update_pair(sa["elo"], sb["elo"], score_a, K_OVERALL)

        # Glicko-1: симметричная пара апдейтов с использованием pre-фактов соперника.
        pre_ra, pre_rda = sa["g_r"], sa["g_rd"]
        pre_rb, pre_rdb = sb["g_r"], sb["g_rd"]
        sa["g_r"], sa["g_rd"] = glicko_update(pre_ra, pre_rda, pre_rb, pre_rdb, score_a)
        sb["g_r"], sb["g_rd"] = glicko_update(pre_rb, pre_rdb, pre_ra, pre_rda, 1.0 - score_a)

        # Striking Elo: победитель - кто нанёс больше значимых ударов.
        a_sig = _safe_float(row.get("f_1_sig_strikes_succ"))
        b_sig = _safe_float(row.get("f_2_sig_strikes_succ"))
        if a_sig + b_sig > 0:
            if a_sig > b_sig:
                str_score = 1.0
            elif a_sig < b_sig:
                str_score = 0.0
            else:
                str_score = 0.5
            sa["str_elo"], sb["str_elo"] = elo_update_pair(
                sa["str_elo"], sb["str_elo"], str_score, K_SKILL
            )

        # Grappling Elo: композит takedowns + sub_att + control.
        a_grp = grappling_score_row(row, 1)
        b_grp = grappling_score_row(row, 2)
        if a_grp + b_grp > 0:
            if a_grp > b_grp:
                grp_score = 1.0
            elif a_grp < b_grp:
                grp_score = 0.0
            else:
                grp_score = 0.5
            sa["grp_elo"], sb["grp_elo"] = elo_update_pair(
                sa["grp_elo"], sb["grp_elo"], grp_score, K_SKILL
            )

        # KO-power vs durability Elo: обновляется только если бой кончился KO/TKO.
        if result_is_ko_tko(row.get("result")):
            ko_score = 1.0 if score_a > 0.5 else 0.0
            sa["ko_elo"], sb["ko_elo"] = elo_update_pair(
                sa["ko_elo"], sb["ko_elo"], ko_score, K_SKILL
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-position cumulative offense / defense
# ---------------------------------------------------------------------------


def _make_long_position_history(fights: pd.DataFrame) -> pd.DataFrame:
    """Длинная таблица один-боец-один-бой с собственными и opp position-stats."""
    frames = []
    for side, opp in [(1, 2), (2, 1)]:
        rows = pd.DataFrame(
            {
                "event_date": fights["event_date"],
                "fight_url": fights["fight_url"],
                "fighter_id": fights[f"f_{side}_id"],
                "fight_duration_sec": fights["fight_duration_sec"],
            }
        )
        for stat in POSITION_STATS:
            rows[stat] = round_stat_sum(fights, side, stat)
            rows[f"opp_{stat}"] = round_stat_sum(fights, opp, stat)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def build_position_history(fights: pd.DataFrame) -> pd.DataFrame:
    """Per-fighter event-level pre-fight position offense + defense (shifted)."""
    long = _make_long_position_history(fights)

    agg_spec: dict[str, tuple[str, str]] = {
        "fight_count": ("fight_url", "count"),
        "fight_duration_sec": ("fight_duration_sec", "sum"),
    }
    for stat in POSITION_STATS:
        agg_spec[stat] = (stat, "sum")
        agg_spec[f"opp_{stat}"] = (f"opp_{stat}", "sum")

    event_level = (
        long.groupby(["fighter_id", "event_date"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["fighter_id", "event_date"])
    )

    grouped = event_level.groupby("fighter_id", group_keys=False)
    sum_cols = (
        ["fight_duration_sec"]
        + list(POSITION_STATS)
        + [f"opp_{stat}" for stat in POSITION_STATS]
    )
    for col in sum_cols:
        event_level[f"cum_{col}"] = grouped[col].cumsum() - event_level[col]

    duration_min = (event_level["cum_fight_duration_sec"] / 60.0).replace(0, np.nan)

    for pos in POSITIONS:
        succ = event_level[f"cum_{pos}_succ"].fillna(0.0)
        att = event_level[f"cum_{pos}_att"].fillna(0.0)
        opp_succ = event_level[f"cum_opp_{pos}_succ"].fillna(0.0)
        opp_att = event_level[f"cum_opp_{pos}_att"].fillna(0.0)

        event_level[f"avg_{pos}_landed_per_min"] = safe_rate(succ, duration_min)
        event_level[f"avg_{pos}_attempted_per_min"] = safe_rate(att, duration_min)
        event_level[f"avg_{pos}_accuracy"] = safe_rate(succ, att)
        event_level[f"avg_{pos}_absorbed_per_min"] = safe_rate(opp_succ, duration_min)
        event_level[f"avg_{pos}_defense"] = 1.0 - safe_rate(opp_succ, opp_att)

    keep = ["fighter_id", "event_date"]
    for pos in POSITIONS:
        for metric in POSITION_OFFENSE_METRICS + POSITION_DEFENSE_METRICS:
            keep.append(f"avg_{pos}_{metric}")
    return event_level[keep]


def position_feature_names() -> list[str]:
    names = []
    for pos in POSITIONS:
        for metric in POSITION_OFFENSE_METRICS + POSITION_DEFENSE_METRICS:
            names.append(f"avg_{pos}_{metric}")
    return names


# ---------------------------------------------------------------------------
# Frame augmentation
# ---------------------------------------------------------------------------


def attach_ids(frame: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(
        fights[["fight_url", "f_1_id", "f_2_id"]],
        on="fight_url",
        how="left",
    )


def add_position_diffs(
    frame: pd.DataFrame, position_history: pd.DataFrame
) -> pd.DataFrame:
    pos_features = position_feature_names()
    for side in (1, 2):
        renamed = position_history.rename(
            columns={col: f"f_{side}_{col}" for col in pos_features}
        )
        frame = frame.merge(
            renamed,
            how="left",
            left_on=[f"f_{side}_id", "event_date"],
            right_on=["fighter_id", "event_date"],
        ).drop(columns=["fighter_id"])
    for feature in pos_features:
        frame[f"{feature}_diff"] = frame[f"f_1_{feature}"] - frame[f"f_2_{feature}"]
    return frame


def add_rating_diffs(frame: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(ratings, how="left", on="fight_url")
    for feature in RATING_FEATURE_NAMES:
        frame[f"{feature}_diff"] = frame[f"f_1_{feature}"] - frame[f"f_2_{feature}"]
    return frame


def build_rated_frame(fights: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Возвращает (frame, all_features, diff_features)."""
    base_frame = build_feature_frame(fights)
    base_frame = attach_ids(base_frame, fights)

    position_history = build_position_history(fights)
    frame = add_position_diffs(base_frame, position_history)

    ratings = compute_pre_fight_ratings(fights)
    frame = add_rating_diffs(frame, ratings)

    base_features = unique_features(BASE_GROUPS)
    position_diff_features = [f"avg_{pos}_{metric}_diff"
                              for pos in POSITIONS
                              for metric in POSITION_OFFENSE_METRICS + POSITION_DEFENSE_METRICS]
    rating_diff_features = [f"{name}_diff" for name in RATING_FEATURE_NAMES]

    features = base_features + position_diff_features + rating_diff_features
    diff_features = [feat for feat in features if feat.endswith("_diff")]
    return frame, features, diff_features


# ---------------------------------------------------------------------------
# Tuning (узкое переопределение под более широкое пространство ExtraTrees)
# ---------------------------------------------------------------------------


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.005, 20.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "extra_trees":
        bootstrap = trial.suggest_categorical("bootstrap", [False, True])
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 3, 5, 7, 9, 12, 16, 20, 24, 30]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.25, 0.4, 0.6, 0.8, 1.0]
            ),
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
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
            "n_estimators": 600,
            "max_depth": 16,
            "min_samples_leaf": 1,
            "min_samples_split": 16,
            "max_features": 1.0,
            "criterion": "gini",
            "class_weight": "balanced_subsample",
            "bootstrap": True,
            "max_samples": 0.55,
        },
        "linear_svm": {"C": 0.1, "class_weight": None},
    }
    return defaults[model_name]


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
        model_name, study.best_params, train_frame, folds, features, diff_features
    )
    return {
        "best_params": study.best_params,
        "best_value_mean_log_loss": float(study.best_value),
        "rolling_cv": cv,
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Главный прогон
# ---------------------------------------------------------------------------


def run_experiment(trials: int, folds: int, model_names: list[str]) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame, features, diff_features = build_rated_frame(fights)

    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    cv_folds = rolling_temporal_folds(train_frame, n_folds=folds)

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
        artifact_path = ARTIFACT_DIR / f"ufc_rated_{model_name}_calibrated.joblib"
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
                "feature_set": "cumulative_plus_last_fight + position_off_def + ratings",
                "feature_count": len(features),
                "base_group_features": len(unique_features(BASE_GROUPS)),
                "position_diff_features": len(POSITIONS)
                * len(POSITION_OFFENSE_METRICS + POSITION_DEFENSE_METRICS),
                "rating_diff_features": len(RATING_FEATURE_NAMES),
                "data_path": str(DATA_PATH),
                "dataset_rows_with_target": int(len(frame)),
                "dataset_min_date": fights["event_date"].min().strftime("%Y-%m-%d"),
                "dataset_max_date": fights["event_date"].max().strftime("%Y-%m-%d"),
                "calibration_method": "sigmoid",
                **final_training,
            },
        }
        joblib.dump(artifact, artifact_path)

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
            "model_importance": model_importance(holdout["base_model"], features)[:40],
            "artifact_path": str(artifact_path),
        }

    best_cv_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["rolling_cv"]["mean_log_loss"],
    )[0]
    best_holdout_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["holdout"]["metrics"]["log_loss"],
    )[0]
    joblib.dump(
        joblib.load(model_results[best_cv_model_name]["artifact_path"]),
        BEST_CV_MODEL_PATH,
    )
    joblib.dump(
        joblib.load(model_results[best_holdout_model_name]["artifact_path"]),
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

    baseline_metrics_path = ARTIFACT_DIR / "ufc_extra_trees_calibrated_metrics.json"
    baseline_metrics = None
    if baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))

    candidate_results_path = ARTIFACT_DIR / "ufc_candidate_models_results.json"
    candidate_metrics = None
    if candidate_results_path.exists():
        candidate_metrics = json.loads(candidate_results_path.read_text(encoding="utf-8"))

    report = {
        "data_path": str(DATA_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "feature_set": "cumulative_plus_last_fight + position_off_def + ratings",
        "feature_count": len(features),
        "features": features,
        "diff_features": diff_features,
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
            "Все cumulative-агрегаты и per-position offense/defense считаются по shifted history.",
            "Pre-fight ratings (Elo / Glicko / skill / SOS) фиксируются ДО апдейта и записываются как признаки.",
            "Rolling CV: train на старых датах, calibration на следующем блоке, validation на следующем.",
            "Финальный holdout - даты после 80%-го хронологического сплита.",
            "Sigmoid calibration на отдельном temporal calibration-блоке.",
            "Train + calibration строки симметрично дополняются перестановкой бойцов.",
        ],
        "rating_engine": {
            "init_elo": INIT_ELO,
            "k_overall": K_OVERALL,
            "k_skill": K_SKILL,
            "init_glicko_rating": INIT_GLICKO_R,
            "init_glicko_rd": INIT_GLICKO_RD,
            "glicko_rd_floor": GLICKO_RD_FLOOR,
            "skill_outcomes": {
                "striking_elo": "выигрывает тот, у кого больше sig_strikes_succ за бой",
                "grappling_elo": "композит takedowns*3 + sub_att*2 + ctrl_min",
                "ko_elo": "обновляется только если бой кончился KO/TKO; победитель=1, иначе нейтрально",
            },
            "strength_of_schedule": "среднее pre-fight Elo всех прошлых соперников",
        },
        "model_results": model_results,
        "ranked_by_holdout_log_loss": ranked,
        "best_model_by_rolling_cv_log_loss": best_cv_model_name,
        "best_model_by_holdout_log_loss": best_holdout_model_name,
        "best_cv_model_artifact_path": str(BEST_CV_MODEL_PATH),
        "best_holdout_model_artifact_path": str(BEST_HOLDOUT_MODEL_PATH),
        "reference_15_feature_artifact": baseline_metrics,
        "reference_candidate_pipeline": candidate_metrics.get("ranked_by_holdout_log_loss")
        if candidate_metrics
        else None,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UFC models with opponent-adjusted ratings and "
                    "per-position offense/defense features."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=25,
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
    report = run_experiment(
        trials=args.trials, folds=args.folds, model_names=args.models
    )
    summary_keys = [
        "feature_count",
        "rolling_cv_folds",
        "best_model_by_rolling_cv_log_loss",
        "best_model_by_holdout_log_loss",
        "ranked_by_holdout_log_loss",
    ]
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
