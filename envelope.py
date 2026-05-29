"""Stage 5: per-fighter empirical envelope для controllable метрик.

Для каждого бойца считаем P10 / P50 / P90 по его историческим per-fight
rates (не cumulative). Используется для realism-check рекомендаций плана:
"увеличить distance volume до X" должен лежать в envelope бойца.

Также возвращает global percentiles (для бойцов с тонкой историей —
fallback на UFC average) и per-fighter mean — стартовая точка плана.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ufc_advanced_pipeline import (
    POSITION_STATS,
    fight_duration_seconds,
    numeric,
    safe_rate,
)


# controllable метрики, для которых строим envelope.
# имя -> функция(row, side) -> per-fight rate
def _per_fight_rates(fights: pd.DataFrame) -> pd.DataFrame:
    """Длинная таблица per-side per-fight rates для controllable действий."""
    duration_min = (fights["fight_duration_sec"] / 60.0).replace(0, np.nan)
    duration_15 = (fights["fight_duration_sec"] / (15.0 * 60.0)).replace(0, np.nan)

    frames = []
    for side, opp in [(1, 2), (2, 1)]:
        sig_succ = numeric(fights[f"f_{side}_sig_strikes_succ"])
        sig_att = numeric(fights[f"f_{side}_sig_strikes_att"])
        td_att = numeric(fights[f"f_{side}_takedown_att"])
        td_succ = numeric(fights[f"f_{side}_takedown_succ"])
        sub_att = numeric(fights[f"f_{side}_submission_att"])
        ctrl_sec = numeric(fights[f"f_{side}_ctrl_time_sec"])
        kd = numeric(fights[f"f_{side}_knockdowns"])

        # per-position attempts
        pos_attempts = {}
        for pos in ["head", "body", "leg", "distance", "clinch", "ground"]:
            # сумма r1..r5 для конкретной позиции (att)
            cols = [f"f_{side}_r{r}_{pos}_att" for r in range(1, 6)]
            cols = [c for c in cols if c in fights.columns]
            if cols:
                pos_attempts[pos] = sum(numeric(fights[c]).fillna(0) for c in cols)
            else:
                pos_attempts[pos] = pd.Series(np.nan, index=fights.index)

        df = pd.DataFrame(
            {
                "fight_url": fights["fight_url"],
                "event_date": fights["event_date"],
                "fighter_id": fights[f"f_{side}_id"],
                "sig_landed_per_min": safe_rate(sig_succ, duration_min),
                "sig_attempted_per_min": safe_rate(sig_att, duration_min),
                "takedowns_attempted_per_15": safe_rate(td_att * 15.0, duration_min),
                "takedowns_landed_per_15": safe_rate(td_succ * 15.0, duration_min),
                "submissions_attempted_per_15": safe_rate(sub_att * 15.0, duration_min),
                "control_minutes_per_15": safe_rate(ctrl_sec / 60.0, duration_15),
                "knockdowns_per_15": safe_rate(kd * 15.0, duration_min),
                "head_attempted_per_min": safe_rate(pos_attempts["head"], duration_min),
                "body_attempted_per_min": safe_rate(pos_attempts["body"], duration_min),
                "leg_attempted_per_min": safe_rate(pos_attempts["leg"], duration_min),
                "distance_attempted_per_min": safe_rate(pos_attempts["distance"], duration_min),
                "clinch_attempted_per_min": safe_rate(pos_attempts["clinch"], duration_min),
                "ground_attempted_per_min": safe_rate(pos_attempts["ground"], duration_min),
                "head_share": safe_rate(
                    pos_attempts["head"],
                    pos_attempts["head"] + pos_attempts["body"] + pos_attempts["leg"],
                ),
                "body_share": safe_rate(
                    pos_attempts["body"],
                    pos_attempts["head"] + pos_attempts["body"] + pos_attempts["leg"],
                ),
                "leg_share": safe_rate(
                    pos_attempts["leg"],
                    pos_attempts["head"] + pos_attempts["body"] + pos_attempts["leg"],
                ),
                "distance_share": safe_rate(
                    pos_attempts["distance"],
                    pos_attempts["distance"] + pos_attempts["clinch"] + pos_attempts["ground"],
                ),
                "clinch_share": safe_rate(
                    pos_attempts["clinch"],
                    pos_attempts["distance"] + pos_attempts["clinch"] + pos_attempts["ground"],
                ),
                "ground_share": safe_rate(
                    pos_attempts["ground"],
                    pos_attempts["distance"] + pos_attempts["clinch"] + pos_attempts["ground"],
                ),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


CONTROLLABLE_METRICS = [
    "sig_landed_per_min",
    "sig_attempted_per_min",
    "takedowns_attempted_per_15",
    "takedowns_landed_per_15",
    "submissions_attempted_per_15",
    "control_minutes_per_15",
    "knockdowns_per_15",
    "head_attempted_per_min",
    "body_attempted_per_min",
    "leg_attempted_per_min",
    "distance_attempted_per_min",
    "clinch_attempted_per_min",
    "ground_attempted_per_min",
    "head_share",
    "body_share",
    "leg_share",
    "distance_share",
    "clinch_share",
    "ground_share",
]


def build_envelope(fights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Возвращает (per_fighter_envelope, global_envelope).

    per_fighter: для каждого (fighter_id, metric) — p10, p50, p90, n.
    global: для каждой metric — p10, p50, p90 (UFC-wide).
    """
    long = _per_fight_rates(fights)

    def agg_percentiles(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if len(s) == 0:
            return pd.Series({"p10": np.nan, "p50": np.nan, "p90": np.nan, "n": 0})
        return pd.Series(
            {
                "p10": float(s.quantile(0.1)),
                "p50": float(s.quantile(0.5)),
                "p90": float(s.quantile(0.9)),
                "n": int(len(s)),
            }
        )

    per_fighter_rows = []
    for fid, group in long.groupby("fighter_id"):
        for metric in CONTROLLABLE_METRICS:
            stats = agg_percentiles(group[metric])
            per_fighter_rows.append({"fighter_id": fid, "metric": metric, **stats.to_dict()})
    per_fighter = pd.DataFrame(per_fighter_rows)

    global_env: dict[str, dict[str, float]] = {}
    for metric in CONTROLLABLE_METRICS:
        stats = agg_percentiles(long[metric])
        global_env[metric] = stats.to_dict()
    return per_fighter, global_env


def within_envelope(
    per_fighter: pd.DataFrame, global_env: dict[str, dict[str, float]],
    fighter_id: str, metric: str, value: float, min_n: int = 3
) -> dict[str, Any]:
    """Возвращает dict с p10/p50/p90 и in_envelope=bool.

    Если у бойца < min_n боёв с этой метрикой — fallback на global.
    """
    row = per_fighter[(per_fighter["fighter_id"] == fighter_id) & (per_fighter["metric"] == metric)]
    if len(row) and int(row.iloc[0]["n"]) >= min_n:
        p10, p50, p90, n = (
            float(row.iloc[0]["p10"]),
            float(row.iloc[0]["p50"]),
            float(row.iloc[0]["p90"]),
            int(row.iloc[0]["n"]),
        )
        source = "fighter"
    else:
        env = global_env.get(metric, {})
        p10, p50, p90, n = (
            float(env.get("p10", np.nan)),
            float(env.get("p50", np.nan)),
            float(env.get("p90", np.nan)),
            int(env.get("n", 0)),
        )
        source = "global"
    return {
        "p10": p10, "p50": p50, "p90": p90, "n": n, "source": source,
        "in_envelope": (p10 <= value <= p90) if not np.isnan(p10 + p90) else False,
    }


def fighter_envelope_summary(
    per_fighter: pd.DataFrame, fighter_id: str
) -> dict[str, dict[str, float]]:
    sub = per_fighter[per_fighter["fighter_id"] == fighter_id]
    return {
        row["metric"]: {"p10": row["p10"], "p50": row["p50"], "p90": row["p90"], "n": int(row["n"])}
        for _, row in sub.iterrows()
    }
