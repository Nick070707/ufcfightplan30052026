"""Каноничный порядок tactical-signature фич + предобработка для кластеризации.

Вынесено в отдельный модуль (не в style_clustering), чтобы `log1p_heavy`
имел стабильный `__module__ == "style_transform"` и сохранённый в пайплайне
FunctionTransformer корректно депикливался в любом процессе (CLI, fight_plan,
streamlit). Если бы функция жила в style_clustering и тот запускался как
`python style_clustering.py`, она бы запиклилась как `__main__.log1p_heavy`
и не загружалась бы в других модулях.
"""

from __future__ import annotations

import numpy as np


# Признаки tactical signature — на них кластеризуем. Все из cumulative_style,
# строго pre-fight, стабильные.
SIGNATURE_FEATURES = [
    "avg_sig_strikes_landed_per_min",
    "avg_sig_strikes_absorbed_per_min",
    "avg_striking_accuracy",
    "avg_striking_defense",
    "avg_takedowns_attempted_per_15",
    "avg_takedown_accuracy",
    "avg_takedown_defense",
    "avg_submissions_attempted_per_15",
    "avg_control_minutes_per_15",
    "avg_knockdowns_per_15",
    "avg_head_strike_share",
    "avg_body_strike_share",
    "avg_leg_strike_share",
    "avg_distance_strike_share",
    "avg_clinch_strike_share",
    "avg_ground_strike_share",
    "avg_fight_duration_min",
]

# Тяжелохвостые rate-фичи: горстка экстремальных бойцов (по сабмишенам/нокдаунам/
# контролю) иначе перетягивает евклидово расстояние и съедает кластеры под себя
# (вырожденные кластеры по 4-8 человек). log1p сжимает хвосты → сбалансированные,
# осмысленные кластеры. Bounded-фичи (accuracy/defense/shares ∈ [0,1]) не трогаем.
HEAVY_TAIL_FEATURES = [
    "avg_sig_strikes_landed_per_min",
    "avg_sig_strikes_absorbed_per_min",
    "avg_takedowns_attempted_per_15",
    "avg_submissions_attempted_per_15",
    "avg_control_minutes_per_15",
    "avg_knockdowns_per_15",
]

_HEAVY_IDX = [SIGNATURE_FEATURES.index(f) for f in HEAVY_TAIL_FEATURES]


def log1p_heavy(X: np.ndarray) -> np.ndarray:
    """log1p только по тяжелохвостым колонкам (порядок = SIGNATURE_FEATURES)."""
    X = np.asarray(X, dtype=float).copy()
    X[:, _HEAVY_IDX] = np.log1p(np.clip(X[:, _HEAVY_IDX], 0.0, None))
    return X
