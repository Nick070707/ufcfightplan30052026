"""UFC decay pipeline.

Шаг поверх ufc_final_pipeline.py (92 признака). Добавляем 8 новых diff-фич:

  Группа D. EWMA-decay аналоги cumulative (5):
    EWMA с halflife=5 боёв для топ-5 cumulative_style метрик. Не замена,
    а дополнение к cumulative — модель сможет учиться на расхождении
    "long-run vs recent form".

  Группа E. Elo-residuals (3):
    Среднее (actual_score − expected_from_elo) по прошлой истории. Ловит
    систематический над- или под-перформанс относительно Elo. Считается
    для overall / striking / grappling Elo.

Все 8 фич — плотные, NaN ~26% (как у cumulative — только дебютанты).
HGB и LR гипотетически оба должны выиграть.

Запуск:
    python ufc_decay_pipeline.py --trials 30 --folds 4
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
    load_clean_fights,
    symmetric_training_set,
)
from ufc_advanced_pipeline import RANDOM_STATE, numeric, safe_rate
from ufc_final_pipeline import (
    build_final_frame,
    default_params,
    evaluate_params_on_folds,
    holdout_evaluation,
    suggest_params,
    train_final_artifact,
)
from ufc_rated_pipeline import (
    INIT_ELO,
    K_OVERALL,
    K_SKILL,
    _safe_float,
    elo_expected,
    elo_update_pair,
    grappling_score_row,
    result_is_ko_tko,
)
from train_candidate_models import (
    chronological_split,
    model_importance,
    rolling_temporal_folds,
)


RESULTS_PATH = ARTIFACT_DIR / "ufc_decay_pipeline_results.json"
BEST_CV_MODEL_PATH = ARTIFACT_DIR / "ufc_decay_best_cv_calibrated.joblib"
BEST_HOLDOUT_MODEL_PATH = ARTIFACT_DIR / "ufc_decay_best_holdout_calibrated.joblib"

MODEL_NAMES = ["logistic_regression", "hist_gradient_boosting"]

EWMA_HALFLIFE = 5.0


# ---------------------------------------------------------------------------
# EWMA history
# ---------------------------------------------------------------------------

# Per-fight rates, считаются из whole-fight stats:
EWMA_METRICS = [
    "ewma_sig_landed_per_min",
    "ewma_striking_defense",
    "ewma_takedowns_landed_per_15",
    "ewma_takedown_defense",
    "ewma_knockdowns_per_15",
]


def _per_fight_rates(fights: pd.DataFrame) -> pd.DataFrame:
    """Длинная per-side таблица с per-fight rate'ами для EWMA."""
    frames = []
    duration_min = (fights["fight_duration_sec"] / 60.0).replace(0, np.nan)
    for side, opp in [(1, 2), (2, 1)]:
        sig_succ = numeric(fights[f"f_{side}_sig_strikes_succ"])
        opp_sig_succ = numeric(fights[f"f_{opp}_sig_strikes_succ"])
        opp_sig_att = numeric(fights[f"f_{opp}_sig_strikes_att"])
        td_succ = numeric(fights[f"f_{side}_takedown_succ"])
        opp_td_succ = numeric(fights[f"f_{opp}_takedown_succ"])
        opp_td_att = numeric(fights[f"f_{opp}_takedown_att"])
        kd = numeric(fights[f"f_{side}_knockdowns"])

        df = pd.DataFrame(
            {
                "fight_url": fights["fight_url"],
                "event_date": fights["event_date"],
                "fighter_id": fights[f"f_{side}_id"],
                "ewma_sig_landed_per_min": safe_rate(sig_succ, duration_min),
                # per-fight striking defense = 1 - opp_sig_succ / opp_sig_att
                "ewma_striking_defense": 1.0 - safe_rate(opp_sig_succ, opp_sig_att),
                "ewma_takedowns_landed_per_15": safe_rate(
                    td_succ * 15.0, duration_min
                ),
                "ewma_takedown_defense": 1.0 - safe_rate(opp_td_succ, opp_td_att),
                "ewma_knockdowns_per_15": safe_rate(kd * 15.0, duration_min),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_ewma_history(
    fights: pd.DataFrame, halflife: float = EWMA_HALFLIFE
) -> pd.DataFrame:
    """Per-fighter pre-fight EWMA метрик (shifted)."""
    long = _per_fight_rates(fights)
    long = long.sort_values(
        ["fighter_id", "event_date", "fight_url"]
    ).reset_index(drop=True)

    # EWMA per fighter, затем shift(1) -> pre-fight.
    grouped = long.groupby("fighter_id", group_keys=False)
    for col in EWMA_METRICS:
        long[col] = grouped[col].transform(
            lambda s: s.ewm(halflife=halflife, min_periods=1).mean().shift(1)
        )

    # Для (fighter_id, event_date) берём mean по возможным дублям
    # (нормально <=1 строка на пару, но on the same date теоретически возможно).
    out = (
        long.groupby(["fighter_id", "event_date"], as_index=False)[EWMA_METRICS]
        .mean()
    )
    return out


# ---------------------------------------------------------------------------
# Elo-residual engine (расширение rated)
# ---------------------------------------------------------------------------

RESIDUAL_FEATURES = [
    "pre_elo_residual_mean",
    "pre_str_residual_mean",
    "pre_grp_residual_mean",
]


def _new_residual_state() -> dict[str, float]:
    return {
        "elo": INIT_ELO,
        "str_elo": INIT_ELO,
        "grp_elo": INIT_ELO,
        "n": 0.0,
        "n_str": 0.0,
        "n_grp": 0.0,
        "elo_resid_sum": 0.0,
        "str_resid_sum": 0.0,
        "grp_resid_sum": 0.0,
    }


def compute_elo_residuals(fights: pd.DataFrame) -> pd.DataFrame:
    """Прокручивает бои в хронологическом порядке и собирает pre-fight
    Elo-residuals (mean past actual − expected).

    Returns DataFrame с одной строкой на бой и f_{1/2}_<RESIDUAL_FEATURES>.
    """
    sorted_fights = fights.sort_values(
        ["event_date", "event_name", "fight_url"]
    ).reset_index(drop=True)

    state: dict[str, dict[str, float]] = defaultdict(_new_residual_state)
    rows: list[dict[str, Any]] = []

    for _, row in sorted_fights.iterrows():
        a_id = row["f_1_id"]
        b_id = row["f_2_id"]
        sa = state[a_id]
        sb = state[b_id]

        pre = {
            "fight_url": row["fight_url"],
            "f_1_pre_elo_residual_mean": (
                sa["elo_resid_sum"] / sa["n"] if sa["n"] > 0 else np.nan
            ),
            "f_2_pre_elo_residual_mean": (
                sb["elo_resid_sum"] / sb["n"] if sb["n"] > 0 else np.nan
            ),
            "f_1_pre_str_residual_mean": (
                sa["str_resid_sum"] / sa["n_str"] if sa["n_str"] > 0 else np.nan
            ),
            "f_2_pre_str_residual_mean": (
                sb["str_resid_sum"] / sb["n_str"] if sb["n_str"] > 0 else np.nan
            ),
            "f_1_pre_grp_residual_mean": (
                sa["grp_resid_sum"] / sa["n_grp"] if sa["n_grp"] > 0 else np.nan
            ),
            "f_2_pre_grp_residual_mean": (
                sb["grp_resid_sum"] / sb["n_grp"] if sb["n_grp"] > 0 else np.nan
            ),
        }
        rows.append(pre)

        score_a = float(row["target_f1_win"])

        # --- Overall Elo residual: actual − expected (pre-fight). ---
        expected_a = elo_expected(sa["elo"], sb["elo"])
        sa["elo_resid_sum"] += score_a - expected_a
        sb["elo_resid_sum"] += (1.0 - score_a) - (1.0 - expected_a)
        sa["n"] += 1.0
        sb["n"] += 1.0
        sa["elo"], sb["elo"] = elo_update_pair(
            sa["elo"], sb["elo"], score_a, K_OVERALL
        )

        # --- Striking residual. ---
        a_sig = _safe_float(row.get("f_1_sig_strikes_succ"))
        b_sig = _safe_float(row.get("f_2_sig_strikes_succ"))
        if a_sig + b_sig > 0:
            str_score = 1.0 if a_sig > b_sig else (0.0 if a_sig < b_sig else 0.5)
            expected_str = elo_expected(sa["str_elo"], sb["str_elo"])
            sa["str_resid_sum"] += str_score - expected_str
            sb["str_resid_sum"] += (1.0 - str_score) - (1.0 - expected_str)
            sa["n_str"] += 1.0
            sb["n_str"] += 1.0
            sa["str_elo"], sb["str_elo"] = elo_update_pair(
                sa["str_elo"], sb["str_elo"], str_score, K_SKILL
            )

        # --- Grappling residual. ---
        a_grp = grappling_score_row(row, 1)
        b_grp = grappling_score_row(row, 2)
        if a_grp + b_grp > 0:
            grp_score = 1.0 if a_grp > b_grp else (0.0 if a_grp < b_grp else 0.5)
            expected_grp = elo_expected(sa["grp_elo"], sb["grp_elo"])
            sa["grp_resid_sum"] += grp_score - expected_grp
            sb["grp_resid_sum"] += (1.0 - grp_score) - (1.0 - expected_grp)
            sa["n_grp"] += 1.0
            sb["n_grp"] += 1.0
            sa["grp_elo"], sb["grp_elo"] = elo_update_pair(
                sa["grp_elo"], sb["grp_elo"], grp_score, K_SKILL
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def add_per_side_history(
    frame: pd.DataFrame, history: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    for side in (1, 2):
        renamed = history.rename(
            columns={col: f"f_{side}_{col}" for col in feature_names}
        )
        frame = frame.merge(
            renamed,
            how="left",
            left_on=[f"f_{side}_id", "event_date"],
            right_on=["fighter_id", "event_date"],
        ).drop(columns=["fighter_id"])
    return frame


def add_diffs(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    for feat in feature_names:
        frame[f"{feat}_diff"] = frame[f"f_1_{feat}"] - frame[f"f_2_{feat}"]
    return frame


def build_decay_frame(
    fights: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame, features, diff_features = build_final_frame(fights)

    ewma = build_ewma_history(fights)
    residuals = compute_elo_residuals(fights)

    frame = add_per_side_history(frame, ewma, EWMA_METRICS)
    # Elo-residuals merge by fight_url (per-fight в residuals уже привязан к
    # порядку чтения исходного fight_url — а в нашем frame fight_url есть).
    frame = frame.merge(residuals, on="fight_url", how="left")

    frame = add_diffs(frame, EWMA_METRICS)
    frame = add_diffs(frame, RESIDUAL_FEATURES)

    new_diff_features = (
        [f"{f}_diff" for f in EWMA_METRICS]
        + [f"{f}_diff" for f in RESIDUAL_FEATURES]
    )
    features = list(features) + new_diff_features
    diff_features = list(diff_features) + new_diff_features
    return frame, features, diff_features


# ---------------------------------------------------------------------------
# Tuning loop (reuse final's)
# ---------------------------------------------------------------------------


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
    frame, features, diff_features = build_decay_frame(fights)

    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    cv_folds = rolling_temporal_folds(train_frame, n_folds=folds)

    new_features = (
        [f"{f}_diff" for f in EWMA_METRICS]
        + [f"{f}_diff" for f in RESIDUAL_FEATURES]
    )

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
            ARTIFACT_DIR / f"ufc_decay_{model_name}_sigmoid_calibrated.joblib"
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
                "feature_set": "final 92 + ewma(halflife=5) + elo-residuals",
                "feature_count": len(features),
                "new_features_added": new_features,
                "ewma_halflife_fights": EWMA_HALFLIFE,
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
            "model_importance": importances[:60] if importances else [],
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

    final_results = _load_reference(
        ARTIFACT_DIR / "ufc_final_pipeline_results.json"
    )
    expdmg_results = _load_reference(
        ARTIFACT_DIR / "ufc_expdmg_pipeline_results.json"
    )

    report = {
        "data_path": str(DATA_PATH),
        "target": "target_f1_win = 1 if winner == f_1_name else 0",
        "feature_set": "final 92 + ewma(halflife=5) + elo-residuals",
        "feature_count": len(features),
        "new_features_added": new_features,
        "ewma_halflife_fights": EWMA_HALFLIFE,
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
            "EWMA — pandas.ewm(halflife=5, min_periods=1).mean().shift(1) per fighter.",
            "Elo-residuals: actual − expected фиксируется ДО апдейта Elo, потом state апдейтится.",
            "Rolling CV: train -> calibration -> validation последовательно по датам.",
            "Holdout: даты после 80%-го хронологического сплита.",
            "Sigmoid calibration на отдельном temporal calibration-блоке.",
            "Train + calibration строки симметрично дополняются перестановкой бойцов.",
        ],
        "model_results": model_results,
        "ranked_by_holdout_log_loss": ranked,
        "best_model_by_rolling_cv_log_loss": best_cv_name,
        "best_model_by_holdout_log_loss": best_holdout_name,
        "best_cv_model_artifact_path": str(BEST_CV_MODEL_PATH),
        "best_holdout_model_artifact_path": str(BEST_HOLDOUT_MODEL_PATH),
        "reference_final_pipeline_ranked": (
            final_results.get("ranked_by_holdout_log_loss")
            if final_results
            else None
        ),
        "reference_expdmg_pipeline_ranked": (
            expdmg_results.get("ranked_by_holdout_log_loss")
            if expdmg_results
            else None
        ),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decay pipeline: final 92 + EWMA(halflife=5) + Elo-residuals."
        )
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(
        trials=args.trials, folds=args.folds, model_names=args.models
    )
    summary = {
        "feature_count": report["feature_count"],
        "new_features_added": report["new_features_added"],
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
