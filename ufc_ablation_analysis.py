from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
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

from ufc_advanced_pipeline import (
    DATA_PATH,
    ARTIFACT_DIR,
    FIGHT_STATS,
    POSITION_STATS,
    RANDOM_STATE,
    fighter_id,
    fight_duration_seconds,
    make_long_history,
    numeric,
    safe_rate,
)


RESULTS_PATH = ARTIFACT_DIR / "ufc_ablation_analysis_results.json"

BASE_FIGHTER_FEATURES = [
    "age_years",
    "height_cm",
    "reach_cm",
]

CUMULATIVE_RESULT_FEATURES = [
    "prior_fights",
    "prior_win_rate",
    "prior_loss_rate",
    "prior_finish_win_rate",
    "prior_finish_loss_rate",
    "prior_ko_loss_rate",
]

CUMULATIVE_STYLE_FEATURES = [
    "avg_fight_duration_min",
    "avg_sig_strikes_landed_per_min",
    "avg_sig_strikes_attempted_per_min",
    "avg_sig_strikes_absorbed_per_min",
    "avg_total_strikes_landed_per_min",
    "avg_total_strikes_attempted_per_min",
    "avg_striking_accuracy",
    "avg_striking_defense",
    "avg_takedowns_landed_per_15",
    "avg_takedowns_attempted_per_15",
    "avg_takedown_accuracy",
    "avg_takedown_defense",
    "avg_submissions_attempted_per_15",
    "avg_control_minutes_per_15",
    "avg_knockdowns_per_15",
    "avg_knockdowns_absorbed_per_15",
    "avg_head_strike_share",
    "avg_body_strike_share",
    "avg_leg_strike_share",
    "avg_distance_strike_share",
    "avg_clinch_strike_share",
    "avg_ground_strike_share",
]

LAST_FIGHT_FEATURES = [
    "has_prior_fight",
    "has_3_prior_fights",
    "days_since_last_fight",
    "days_since_last_fight_missing",
    "last_fight_duration_min",
    "last_fight_win",
    "last_fight_loss",
    "last_fight_ko_loss",
]

RECENT_FORM_FEATURES = [
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

CONTEXT_FEATURES = [
    "same_stance",
    "title_fight",
    "scheduled_rounds",
    "female_division",
]


def diff_names(features: list[str]) -> list[str]:
    return [f"{feature}_diff" for feature in features]


FEATURE_GROUPS = {
    "physical": diff_names(BASE_FIGHTER_FEATURES),
    "cumulative_results": diff_names(CUMULATIVE_RESULT_FEATURES),
    "cumulative_style": diff_names(CUMULATIVE_STYLE_FEATURES),
    "last_fight_state": diff_names(LAST_FIGHT_FEATURES),
    "recent_form": diff_names(RECENT_FORM_FEATURES),
    "context": CONTEXT_FEATURES,
}


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


def rolling_mean_by_fighter(grouped: pd.core.groupby.DataFrameGroupBy, column: str, window: int) -> pd.Series:
    return grouped[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).mean()
    )


def ewma_by_fighter(grouped: pd.core.groupby.DataFrameGroupBy, column: str, alpha: float = 0.45) -> pd.Series:
    return grouped[column].transform(
        lambda values: values.shift(1).ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    )


def build_all_history(df: pd.DataFrame) -> pd.DataFrame:
    long_history = make_long_history(df)
    rate_sum_columns = ["win", "loss", "finish_win", "finish_loss", "ko_loss"]
    stat_sum_columns = [
        "fight_duration_sec",
        *FIGHT_STATS,
        *[f"opp_{stat}" for stat in FIGHT_STATS],
        *POSITION_STATS,
    ]
    event_level = (
        long_history.groupby(["fighter_id", "event_date"], as_index=False)
        .agg(
            fight_count=("fight_url", "count"),
            win=("win", "mean"),
            loss=("loss", "mean"),
            finish_win=("finish_win", "mean"),
            finish_loss=("finish_loss", "mean"),
            ko_loss=("ko_loss", "mean"),
            fight_duration_sec_current=("fight_duration_sec", "mean"),
            **{f"{col}_sum": (col, "sum") for col in rate_sum_columns},
            **{col: (col, "sum") for col in stat_sum_columns},
        )
        .sort_values(["fighter_id", "event_date"])
    )

    grouped = event_level.groupby("fighter_id", group_keys=False)
    previous_event_date = grouped["event_date"].shift(1)
    event_level["prior_fights"] = grouped["fight_count"].cumsum() - event_level["fight_count"]
    event_level["has_prior_fight"] = event_level["prior_fights"].ge(1).astype(float)
    event_level["has_3_prior_fights"] = event_level["prior_fights"].ge(3).astype(float)
    event_level["days_since_last_fight"] = (event_level["event_date"] - previous_event_date).dt.days
    event_level["days_since_last_fight_missing"] = event_level["days_since_last_fight"].isna().astype(float)
    event_level["last_fight_duration_min"] = grouped["fight_duration_sec_current"].shift(1) / 60.0
    event_level["last_fight_win"] = grouped["win"].shift(1)
    event_level["last_fight_loss"] = grouped["loss"].shift(1)
    event_level["last_fight_ko_loss"] = grouped["ko_loss"].shift(1)

    cumulative_sum_columns = [
        "fight_count",
        *[f"{col}_sum" for col in rate_sum_columns],
        *stat_sum_columns,
    ]
    for col in cumulative_sum_columns:
        event_level[f"cum_{col}"] = grouped[col].cumsum() - event_level[col]

    fights = event_level["prior_fights"].replace(0, np.nan)
    duration_sec = event_level["cum_fight_duration_sec"].fillna(0.0)
    duration_min = (duration_sec / 60.0).replace(0, np.nan)
    duration_15 = (duration_sec / (15.0 * 60.0)).replace(0, np.nan)
    sig_att = event_level["cum_sig_strikes_att"].fillna(0.0)
    opp_sig_att = event_level["cum_opp_sig_strikes_att"].fillna(0.0)
    td_att = event_level["cum_takedown_att"].fillna(0.0)
    opp_td_att = event_level["cum_opp_takedown_att"].fillna(0.0)
    position_att = (
        event_level["cum_head_att"].fillna(0.0)
        + event_level["cum_body_att"].fillna(0.0)
        + event_level["cum_leg_att"].fillna(0.0)
    )
    range_att = (
        event_level["cum_distance_att"].fillna(0.0)
        + event_level["cum_clinch_att"].fillna(0.0)
        + event_level["cum_ground_att"].fillna(0.0)
    )

    event_level["prior_win_rate"] = safe_rate(event_level["cum_win_sum"].fillna(0.0), fights)
    event_level["prior_loss_rate"] = safe_rate(event_level["cum_loss_sum"].fillna(0.0), fights)
    event_level["prior_finish_win_rate"] = safe_rate(event_level["cum_finish_win_sum"].fillna(0.0), fights)
    event_level["prior_finish_loss_rate"] = safe_rate(event_level["cum_finish_loss_sum"].fillna(0.0), fights)
    event_level["prior_ko_loss_rate"] = safe_rate(event_level["cum_ko_loss_sum"].fillna(0.0), fights)
    event_level["avg_fight_duration_min"] = safe_rate(duration_sec / 60.0, fights)
    event_level["avg_sig_strikes_landed_per_min"] = safe_rate(
        event_level["cum_sig_strikes_succ"].fillna(0.0), duration_min
    )
    event_level["avg_sig_strikes_attempted_per_min"] = safe_rate(sig_att, duration_min)
    event_level["avg_sig_strikes_absorbed_per_min"] = safe_rate(
        event_level["cum_opp_sig_strikes_succ"].fillna(0.0), duration_min
    )
    event_level["avg_total_strikes_landed_per_min"] = safe_rate(
        event_level["cum_total_strikes_succ"].fillna(0.0), duration_min
    )
    event_level["avg_total_strikes_attempted_per_min"] = safe_rate(
        event_level["cum_total_strikes_att"].fillna(0.0), duration_min
    )
    event_level["avg_striking_accuracy"] = safe_rate(
        event_level["cum_sig_strikes_succ"].fillna(0.0), sig_att
    )
    event_level["avg_striking_defense"] = 1.0 - safe_rate(
        event_level["cum_opp_sig_strikes_succ"].fillna(0.0), opp_sig_att
    )
    event_level["avg_takedowns_landed_per_15"] = safe_rate(
        event_level["cum_takedown_succ"].fillna(0.0), duration_15
    )
    event_level["avg_takedowns_attempted_per_15"] = safe_rate(td_att, duration_15)
    event_level["avg_takedown_accuracy"] = safe_rate(
        event_level["cum_takedown_succ"].fillna(0.0), td_att
    )
    event_level["avg_takedown_defense"] = 1.0 - safe_rate(
        event_level["cum_opp_takedown_succ"].fillna(0.0), opp_td_att
    )
    event_level["avg_submissions_attempted_per_15"] = safe_rate(
        event_level["cum_submission_att"].fillna(0.0), duration_15
    )
    event_level["avg_control_minutes_per_15"] = safe_rate(
        event_level["cum_ctrl_time_sec"].fillna(0.0) / 60.0, duration_15
    )
    event_level["avg_knockdowns_per_15"] = safe_rate(
        event_level["cum_knockdowns"].fillna(0.0), duration_15
    )
    event_level["avg_knockdowns_absorbed_per_15"] = safe_rate(
        event_level["cum_opp_knockdowns"].fillna(0.0), duration_15
    )
    event_level["avg_head_strike_share"] = safe_rate(event_level["cum_head_att"].fillna(0.0), position_att)
    event_level["avg_body_strike_share"] = safe_rate(event_level["cum_body_att"].fillna(0.0), position_att)
    event_level["avg_leg_strike_share"] = safe_rate(event_level["cum_leg_att"].fillna(0.0), position_att)
    event_level["avg_distance_strike_share"] = safe_rate(
        event_level["cum_distance_att"].fillna(0.0), range_att
    )
    event_level["avg_clinch_strike_share"] = safe_rate(
        event_level["cum_clinch_att"].fillna(0.0), range_att
    )
    event_level["avg_ground_strike_share"] = safe_rate(
        event_level["cum_ground_att"].fillna(0.0), range_att
    )

    current_duration_min = (event_level["fight_duration_sec_current"] / 60.0).replace(0, np.nan)
    current_duration_15 = (event_level["fight_duration_sec_current"] / (15.0 * 60.0)).replace(0, np.nan)
    event_level["cur_fight_duration_min"] = current_duration_min
    event_level["cur_sig_landed_per_min"] = safe_rate(event_level["sig_strikes_succ"], current_duration_min)
    event_level["cur_sig_absorbed_per_min"] = safe_rate(event_level["opp_sig_strikes_succ"], current_duration_min)
    event_level["cur_takedowns_landed_per_15"] = safe_rate(event_level["takedown_succ"], current_duration_15)
    event_level["cur_takedowns_attempted_per_15"] = safe_rate(event_level["takedown_att"], current_duration_15)
    event_level["cur_submissions_attempted_per_15"] = safe_rate(event_level["submission_att"], current_duration_15)
    event_level["cur_control_minutes_per_15"] = safe_rate(event_level["ctrl_time_sec"] / 60.0, current_duration_15)
    event_level["cur_knockdowns_per_15"] = safe_rate(event_level["knockdowns"], current_duration_15)
    event_level["cur_striking_defense"] = 1.0 - safe_rate(
        event_level["opp_sig_strikes_succ"], event_level["opp_sig_strikes_att"]
    )

    recent_source_columns = {
        "win_rate": "win",
        "ko_loss_rate": "ko_loss",
        "fight_duration_min": "cur_fight_duration_min",
        "sig_landed_per_min": "cur_sig_landed_per_min",
        "sig_absorbed_per_min": "cur_sig_absorbed_per_min",
        "takedowns_landed_per_15": "cur_takedowns_landed_per_15",
        "takedowns_attempted_per_15": "cur_takedowns_attempted_per_15",
        "submissions_attempted_per_15": "cur_submissions_attempted_per_15",
        "control_minutes_per_15": "cur_control_minutes_per_15",
        "knockdowns_per_15": "cur_knockdowns_per_15",
        "striking_defense": "cur_striking_defense",
    }
    for window in [1, 3, 5]:
        for out_name, source_col in recent_source_columns.items():
            if window == 1 and out_name in {"win_rate", "ko_loss_rate", "fight_duration_min"}:
                continue
            event_level[f"recent_{window}_{out_name}"] = rolling_mean_by_fighter(grouped, source_col, window)

    ewma_source_columns = {
        "win_rate": "win",
        "ko_loss_rate": "ko_loss",
        "sig_landed_per_min": "cur_sig_landed_per_min",
        "sig_absorbed_per_min": "cur_sig_absorbed_per_min",
        "takedowns_landed_per_15": "cur_takedowns_landed_per_15",
        "control_minutes_per_15": "cur_control_minutes_per_15",
        "striking_defense": "cur_striking_defense",
    }
    for out_name, source_col in ewma_source_columns.items():
        event_level[f"ewma_{out_name}"] = ewma_by_fighter(grouped, source_col)

    history_features = (
        CUMULATIVE_RESULT_FEATURES
        + CUMULATIVE_STYLE_FEATURES
        + LAST_FIGHT_FEATURES
        + RECENT_FORM_FEATURES
    )
    return event_level[["fighter_id", "event_date", *history_features]]


def add_side_history(base: pd.DataFrame, history: pd.DataFrame, side: int) -> pd.DataFrame:
    history_cols = [col for col in history.columns if col not in {"fighter_id", "event_date"}]
    renamed = history.rename(columns={col: f"f_{side}_{col}" for col in history_cols})
    return base.merge(
        renamed,
        how="left",
        left_on=[f"f_{side}_id", "event_date"],
        right_on=["fighter_id", "event_date"],
    ).drop(columns=["fighter_id"])


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    history = build_all_history(df)
    base = add_side_history(df, history, 1)
    base = add_side_history(base, history, 2)

    for side in [1, 2]:
        prefix = f"f_{side}"
        dob = pd.to_datetime(base[f"{prefix}_fighter_dob"], errors="coerce")
        base[f"{prefix}_age_years"] = (base["event_date"] - dob).dt.days / 365.25
        base[f"{prefix}_height_cm"] = numeric(base[f"{prefix}_fighter_height_cm"])
        base[f"{prefix}_reach_cm"] = numeric(base[f"{prefix}_fighter_reach_cm"])

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

    side_features = (
        BASE_FIGHTER_FEATURES
        + CUMULATIVE_RESULT_FEATURES
        + CUMULATIVE_STYLE_FEATURES
        + LAST_FIGHT_FEATURES
        + RECENT_FORM_FEATURES
    )
    for feature in side_features:
        frame[f"{feature}_diff"] = base[f"f_1_{feature}"] - base[f"f_2_{feature}"]
    return frame


def chronological_split(frame: pd.DataFrame, train_date_fraction: float = 0.8):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * train_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    train_mask = frame["event_date"].le(cutoff_date)
    test_mask = frame["event_date"].gt(cutoff_date)
    return train_mask, test_mask, cutoff_date


def temporal_validation_split(frame: pd.DataFrame, fit_date_fraction: float = 0.85):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * fit_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    fit_mask = frame["event_date"].le(cutoff_date)
    validation_mask = frame["event_date"].gt(cutoff_date)
    return fit_mask, validation_mask, cutoff_date


def symmetric_training_set(X: pd.DataFrame, y: pd.Series, diff_features: list[str]):
    swapped = X.copy()
    swapped[diff_features] = -swapped[diff_features]
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


def make_model(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.2,
                        max_iter=4000,
                        random_state=RANDOM_STATE,
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
                        n_estimators=350,
                        max_depth=8,
                        min_samples_leaf=18,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        min_samples_leaf=35,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def unique_features(groups: list[str]) -> list[str]:
    features: list[str] = []
    seen = set()
    for group in groups:
        for feature in FEATURE_GROUPS[group]:
            if feature not in seen:
                seen.add(feature)
                features.append(feature)
    return features


def all_feature_columns() -> list[str]:
    return unique_features(["physical", "cumulative_results", "cumulative_style", "last_fight_state", "recent_form", "context"])


def select_univariate_features(X_fit: pd.DataFrame, y_fit: pd.Series, k: int) -> list[str]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_fit)
    scores, p_values = f_classif(X_imp, y_fit)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)[::-1]
    return X_fit.columns[order[: min(k, len(order))]].tolist()


def select_tree_features(X_fit: pd.DataFrame, y_fit: pd.Series, k: int) -> list[str]:
    model = make_model("extra_trees")
    model.fit(X_fit, y_fit)
    importances = model.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)[::-1]
    return X_fit.columns[order[: min(k, len(order))]].tolist()


def evaluate_feature_set(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    fit_mask: pd.Series,
    validation_mask: pd.Series,
    features: list[str],
    model_names: list[str],
) -> dict[str, Any]:
    diff_features = [feature for feature in features if feature.endswith("_diff")]
    X_fit, y_fit = symmetric_training_set(
        train_frame.loc[fit_mask, features],
        train_frame.loc[fit_mask, "target_f1_win"],
        diff_features,
    )
    X_valid, y_valid = symmetric_training_set(
        train_frame.loc[validation_mask, features],
        train_frame.loc[validation_mask, "target_f1_win"],
        diff_features,
    )
    X_train, y_train = symmetric_training_set(
        train_frame[features],
        train_frame["target_f1_win"],
        diff_features,
    )
    X_test = test_frame[features]
    y_test = test_frame["target_f1_win"]

    model_results = {}
    for model_name in model_names:
        validation_model = make_model(model_name)
        validation_model.fit(X_fit, y_fit)
        validation_probabilities = validation_model.predict_proba(X_valid)[:, 1]

        test_model = make_model(model_name)
        test_model.fit(X_train, y_train)
        test_probabilities = test_model.predict_proba(X_test)[:, 1]

        model_results[model_name] = {
            "validation_metrics": metric_block(y_valid, validation_probabilities),
            "test_metrics": metric_block(y_test, test_probabilities),
        }

    best_model_name = min(
        model_results.items(),
        key=lambda item: item[1]["validation_metrics"]["log_loss"],
    )[0]
    return {
        "feature_count": len(features),
        "features": features,
        "models": model_results,
        "best_model_by_validation_log_loss": best_model_name,
        "best_test_metrics": model_results[best_model_name]["test_metrics"],
    }


def run_ablation(model_names: list[str], selection_sizes: list[int]) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    fights = load_clean_fights(DATA_PATH)
    frame = build_feature_frame(fights)
    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    fit_mask, validation_mask, validation_cutoff = temporal_validation_split(train_frame)

    group_recipes = {
        "physical_context": ["physical", "context"],
        "physical_cumulative_results": ["physical", "context", "cumulative_results"],
        "physical_cumulative_style": ["physical", "context", "cumulative_style"],
        "physical_last_fight": ["physical", "context", "last_fight_state"],
        "physical_recent_form": ["physical", "context", "recent_form"],
        "cumulative_all": ["physical", "context", "cumulative_results", "cumulative_style"],
        "cumulative_plus_last_fight": [
            "physical",
            "context",
            "cumulative_results",
            "cumulative_style",
            "last_fight_state",
        ],
        "cumulative_plus_recent": [
            "physical",
            "context",
            "cumulative_results",
            "cumulative_style",
            "recent_form",
        ],
        "all_features": [
            "physical",
            "context",
            "cumulative_results",
            "cumulative_style",
            "last_fight_state",
            "recent_form",
        ],
    }

    ablation_results = {}
    for name, groups in group_recipes.items():
        ablation_results[name] = evaluate_feature_set(
            frame,
            train_frame,
            test_frame,
            fit_mask,
            validation_mask,
            unique_features(groups),
            model_names,
        )

    all_features = all_feature_columns()
    all_diff_features = [feature for feature in all_features if feature.endswith("_diff")]
    X_fit_all, y_fit_all = symmetric_training_set(
        train_frame.loc[fit_mask, all_features],
        train_frame.loc[fit_mask, "target_f1_win"],
        all_diff_features,
    )

    selection_results = {}
    for k in selection_sizes:
        univariate = select_univariate_features(X_fit_all, y_fit_all, k)
        selection_results[f"univariate_f_score_top_{k}"] = evaluate_feature_set(
            frame,
            train_frame,
            test_frame,
            fit_mask,
            validation_mask,
            univariate,
            model_names,
        )
        tree_selected = select_tree_features(X_fit_all, y_fit_all, k)
        selection_results[f"extra_trees_importance_top_{k}"] = evaluate_feature_set(
            frame,
            train_frame,
            test_frame,
            fit_mask,
            validation_mask,
            tree_selected,
            model_names,
        )

    combined_results = {**ablation_results, **selection_results}
    ranked = sorted(
        [
            {
                "name": name,
                "feature_count": result["feature_count"],
                "best_model": result["best_model_by_validation_log_loss"],
                **result["best_test_metrics"],
            }
            for name, result in combined_results.items()
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
        "feature_group_counts": {group: len(features) for group, features in FEATURE_GROUPS.items()},
        "all_feature_count": len(all_features),
        "model_names": model_names,
        "selection_sizes": selection_sizes,
        "rows_total_with_target": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "validation_cutoff_fit_lte": validation_cutoff.strftime("%Y-%m-%d"),
        "test_date_min": test_frame["event_date"].min().strftime("%Y-%m-%d"),
        "test_date_max": test_frame["event_date"].max().strftime("%Y-%m-%d"),
        "leakage_policy": [
            "All fighter history features are shifted by event_date before joining to the current fight.",
            "Cumulative, last-fight, rolling, and EWMA features use only prior events.",
            "Current-fight winner/result/finish fields, current aggregate stats, odds, and rankings are not model features.",
            "Mutable UFCStats career profile performance fields are excluded.",
            "Training rows are symmetrically augmented with swapped fighters.",
        ],
        "ablation_results": ablation_results,
        "feature_selection_results": selection_results,
        "ranked_by_test_log_loss": ranked,
        "reference_15_feature_artifact": baseline_metrics,
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFC feature-group ablation and feature selection.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "extra_trees", "hist_gradient_boosting"],
        choices=["logistic_regression", "extra_trees", "hist_gradient_boosting"],
    )
    parser.add_argument(
        "--selection-sizes",
        nargs="+",
        type=int,
        default=[15, 25, 35, 45],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_ablation(model_names=args.models, selection_sizes=args.selection_sizes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
