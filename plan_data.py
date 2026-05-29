"""Shared utilities for MVP fight plan generator.

Кеширует frame из ufc_decay_pipeline, чтобы style_clustering /
asymmetry_diagnostics / envelope не пересобирали 100 фич каждый раз.
Также извлекает per-fighter snapshot и предсказывает P(win).
"""

from __future__ import annotations

import tempfile
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ufc_ablation_analysis import (
    DATA_PATH,
    build_all_history,
    load_clean_fights,
)
from ufc_advanced_pipeline import FIGHT_STATS, POSITION_STATS
from ufc_decay_pipeline import build_decay_frame


# Артефакты модели лежат в репо (read-only на cloud — но это OK, мы их только читаем).
ARTIFACT_DIR = Path("artifacts")
HGB_ARTIFACT = ARTIFACT_DIR / "ufc_decay_hist_gradient_boosting_sigmoid_calibrated.joblib"


def _pick_cache_dir() -> Path:
    """ARTIFACT_DIR если writable (локально), иначе tempdir (Streamlit Cloud).
    Решается один раз при импорте модуля."""
    try:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        probe = ARTIFACT_DIR / "._probe_writable"
        probe.write_bytes(b"")
        probe.unlink()
        return ARTIFACT_DIR
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ufc_plan_cache"
        fallback.mkdir(exist_ok=True, parents=True)
        return fallback


# v2: снапшот строится на синтетическом «следующем бое» (включает последний
# реальный бой). Версия в имени кеша инвалидирует старые (отстающие) кеши.
CACHE_DIR = _pick_cache_dir()
FRAME_CACHE = CACHE_DIR / "_plan_decay_frame_v2.parquet"
FRAME_META_CACHE = CACHE_DIR / "_plan_decay_meta_v2.joblib"
HISTORY_CACHE = CACHE_DIR / "_plan_decay_history_v2.parquet"
PROFILE_CACHE = CACHE_DIR / "_plan_decay_profile_v2.parquet"

# Синтетический «следующий бой» для свежего снапшота (фикс «отставания на бой»).
SYNTH_OPP_ID = "__SYNTHETIC_OPPONENT__"
SYNTH_OPP_NAME = "__SYNTHETIC_OPPONENT__"
SYNTH_EVENT_NAME = "__SYNTHETIC_SNAPSHOT__"
SYNTH_FIGHT_URL_PREFIX = "__synthetic__/"
# дата синтетического боя = последний бой бойца + это число дней. Держим
# возраст/раскладку близкими к реальным, но последний бой уже включён.
SNAPSHOT_LAYOFF_DAYS = 180


def _synth_stat_columns(columns: pd.Index) -> list[str]:
    """Колонки сырых статов боя (их обнуляем на синтетической строке, чтобы
    cumsum не ловил NaN; в pre-fight фичи они всё равно не входят)."""
    out = ["fight_duration_sec"]
    for side in (1, 2):
        out += [f"f_{side}_{st}" for st in FIGHT_STATS]
        for r in range(1, 6):
            out += [f"f_{side}_r{r}_{ps}" for ps in POSITION_STATS]
    return [c for c in out if c in columns]


def _swap_sides(row: pd.Series) -> pd.Series:
    """Меняет местами все f_1_* и f_2_* поля строки боя (чтобы боец стал f_1)."""
    new = row.copy()
    for col in row.index:
        if col.startswith("f_1_"):
            twin = "f_2_" + col[len("f_1_"):]
            if twin in row.index:
                new[col] = row[twin]
                new[twin] = row[col]
    return new


def _append_synthetic_snapshot_fights(fights: pd.DataFrame) -> pd.DataFrame:
    """Добавляет по одному синтетическому «следующему бою» на каждого бойца.

    Боец ставится в угол f_1, оппонент — общий dummy (`SYNTH_OPP_ID`), дата —
    его последний бой + SNAPSHOT_LAYOFF_DAYS. Прогнанные через тот же
    `build_decay_frame`, pre-fight фичи этой строки = кумулятив по ВСЕЙ реальной
    истории бойца (паттерн `cumsum − current` / `shift(1)` вычитает сам
    синтетический бой). Так снапшот перестаёт «отставать на бой».

    Корректность без перекрёстного загрязнения: dummy — единственный оппонент
    во всех синтетических боях, поэтому каждый реальный боец появляется на
    будущей дате ровно один раз (как f_1); состояние dummy ни на кого не влияет,
    т.к. читается только pre-fight f_1.
    """
    fights = fights.copy()
    fights["is_synthetic"] = False

    # последняя по дате строка каждого бойца + с какой он был стороны
    appearances = []
    for side in (1, 2):
        sub = pd.DataFrame(
            {
                "event_date": fights["event_date"],
                "fighter_id": fights[f"f_{side}_id"],
                "row_idx": fights.index,
                "side": side,
            }
        )
        appearances.append(sub)
    appearances = pd.concat(appearances, ignore_index=True).dropna(
        subset=["fighter_id"]
    )
    last = (
        appearances.sort_values("event_date")
        .groupby("fighter_id", as_index=False)
        .tail(1)
    )

    stat_cols = _synth_stat_columns(fights.columns)
    synth_rows = []
    for rec in last.itertuples(index=False):
        if rec.fighter_id == SYNTH_OPP_ID:
            continue
        base = fights.loc[rec.row_idx].copy()
        if rec.side == 2:
            base = _swap_sides(base)
        # теперь боец — f_1; оппонент → dummy
        base["f_2_id"] = SYNTH_OPP_ID
        base["f_2_name"] = SYNTH_OPP_NAME
        if "f_2_fighter_url" in base.index:
            base["f_2_fighter_url"] = SYNTH_OPP_ID
        if "f_2_url" in base.index:
            base["f_2_url"] = SYNTH_OPP_ID
        for col in stat_cols:
            base[col] = 0.0
        base["event_date"] = rec.event_date + pd.Timedelta(days=SNAPSHOT_LAYOFF_DAYS)
        base["event_name"] = SYNTH_EVENT_NAME
        base["fight_url"] = f"{SYNTH_FIGHT_URL_PREFIX}{rec.fighter_id}"
        base["winner"] = base["f_1_name"]
        base["target_f1_win"] = 1
        base["is_synthetic"] = True
        synth_rows.append(base)

    synth = pd.DataFrame(synth_rows)
    out = pd.concat([fights, synth], ignore_index=True)
    out = out.sort_values(
        ["event_date", "event_name", "fight_url"]
    ).reset_index(drop=True)
    return out


def _per_side_profile_long(fights: pd.DataFrame) -> pd.DataFrame:
    """Длинная таблица: только колонки, которых нет в frame per-side
    (fighter_stance/weight_lbs/reach_cm уже привязаны через matchup-фичи)."""
    profile_cols = ["fighter_height_cm", "fighter_dob"]
    rows = []
    for side in (1, 2):
        cols = ["event_date", f"f_{side}_id"]
        cols += [f"f_{side}_{c}" for c in profile_cols]
        sub = fights[cols].copy()
        sub.columns = ["event_date", "fighter_id"] + profile_cols
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _per_side_layoff(history: pd.DataFrame) -> pd.DataFrame:
    return history[["fighter_id", "event_date", "days_since_last_fight"]].copy()


def _build_frame_fresh() -> tuple[
    pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame
]:
    """Возвращает frame + features + history + profile (per-side raw profile).

    Поверх реальных боёв добавляется по одному синтетическому «следующему бою»
    на бойца — pre-fight фичи такой строки дают свежий снапшот, включающий
    последний реальный бой (см. _append_synthetic_snapshot_fights).
    """
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    fights = load_clean_fights(DATA_PATH)
    fights = _append_synthetic_snapshot_fights(fights)
    frame, features, diff_features = build_decay_frame(fights)
    # build_feature_frame пересобирает frame с нуля — переносим флаг по fight_url
    frame["is_synthetic"] = (
        frame["fight_url"].astype(str).str.startswith(SYNTH_FIGHT_URL_PREFIX)
    )
    history = build_all_history(fights)
    profile = _per_side_profile_long(fights)
    return frame, features, diff_features, history, profile


def load_frame(use_cache: bool = True) -> tuple[
    pd.DataFrame, list[str], list[str], pd.DataFrame, pd.DataFrame
]:
    """Возвращает (frame, features, diff_features, history, profile). Кеш в parquet/joblib."""
    if (
        use_cache
        and FRAME_CACHE.exists()
        and FRAME_META_CACHE.exists()
        and HISTORY_CACHE.exists()
        and PROFILE_CACHE.exists()
    ):
        frame = pd.read_parquet(FRAME_CACHE)
        history = pd.read_parquet(HISTORY_CACHE)
        profile = pd.read_parquet(PROFILE_CACHE)
        meta = joblib.load(FRAME_META_CACHE)
        return frame, meta["features"], meta["diff_features"], history, profile

    frame, features, diff_features, history, profile = _build_frame_fresh()
    _try_persist_cache(frame, features, diff_features, history, profile)
    return frame, features, diff_features, history, profile


def _try_persist_cache(frame, features, diff_features, history, profile) -> None:
    """Пишет parquet/joblib кеш в CACHE_DIR (выбран при импорте — writable)."""
    try:
        safe = frame.copy()
        for col in safe.columns:
            if safe[col].dtype == object:
                safe[col] = safe[col].astype("string")
        safe.to_parquet(FRAME_CACHE, index=False)
        history.to_parquet(HISTORY_CACHE, index=False)
        profile.to_parquet(PROFILE_CACHE, index=False)
        joblib.dump(
            {"features": features, "diff_features": diff_features},
            FRAME_META_CACHE,
        )
    except OSError:
        # на всякий случай — если writability изменилась после импорта
        pass


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    return joblib.load(HGB_ARTIFACT)


# ---------------------------------------------------------------------------
# Per-fighter latest snapshot
# ---------------------------------------------------------------------------


def _per_side_features(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("f_1_")
            and c not in {"f_1_name", "f_1_id"}]


# NB: position-shrinkage делается на уровне asymmetry_diagnostics._shrink
# (prior=40 попыток) во время генерации плана. Снапшот хранит сырые avg_*.


def build_fighter_index(
    frame: pd.DataFrame, history: pd.DataFrame, profile: pd.DataFrame
) -> pd.DataFrame:
    """Возвращает per-fighter latest snapshot.

    Каждая строка — один боец (по f_*_id). Колонки:
      fighter_id, fighter_name, event_date (последний бой), num_career_fights,
      все per-side метрики из frame (без префикса) +
      все cumulative/last-fight метрики из history.
    """
    side_metric_cols = [
        c[len("f_1_"):] for c in frame.columns
        if c.startswith("f_1_") and c not in {"f_1_name", "f_1_id"}
    ]
    has_synth = "is_synthetic" in frame.columns

    rows = []
    for side in (1, 2):
        cols = ["event_date", f"f_{side}_id", f"f_{side}_name"]
        cols += [f"f_{side}_{m}" for m in side_metric_cols]
        if has_synth:
            cols += ["is_synthetic"]
        sub = frame[cols].copy()
        out_cols = ["event_date", "fighter_id", "fighter_name"] + side_metric_cols
        if has_synth:
            out_cols += ["is_synthetic"]
        sub.columns = out_cols
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["fighter_id"])
    # dummy-оппонент синтетических боёв — не реальный боец
    long = long[long["fighter_id"] != SYNTH_OPP_ID]
    if not has_synth:
        long["is_synthetic"] = False
    long = long.sort_values(["fighter_id", "event_date"])

    # history содержит recent_form/ewma колонки, пересекающиеся с per-side
    # frame EWMA — дропаем их (модель использует только frame-овые halflife=5).
    drop_overlap = [
        c for c in history.columns
        if c.startswith("recent_") or c.startswith("ewma_")
    ]
    history_trim = history.drop(columns=drop_overlap)
    long = long.merge(history_trim, on=["fighter_id", "event_date"], how="left")
    long = long.merge(profile, on=["fighter_id", "event_date"], how="left")

    dob = pd.to_datetime(long["fighter_dob"], errors="coerce")
    long["age_years"] = (long["event_date"] - dob).dt.days / 365.25
    long["height_cm"] = pd.to_numeric(long["fighter_height_cm"], errors="coerce")
    long["reach_cm"] = pd.to_numeric(long["fighter_reach_cm"], errors="coerce")
    long["weight_lbs"] = pd.to_numeric(long["fighter_weight_lbs"], errors="coerce")
    long["southpaw"] = (
        long["fighter_stance"].fillna("").astype(str).str.lower().eq("southpaw").astype(int)
    )

    # Снапшот = самая поздняя строка бойца. При наличии синтетического боя это
    # именно он → фичи включают последний реальный бой (нет «отставания»).
    latest = long.groupby("fighter_id", as_index=False).tail(1).reset_index(drop=True)
    # карьерные бои считаем только по реальным строкам (синтетический не в счёт)
    fight_counts = (
        long[~long["is_synthetic"].fillna(False)]
        .groupby("fighter_id")
        .size()
        .rename("num_career_fights")
    )
    latest = latest.drop(columns=["is_synthetic"], errors="ignore")
    latest = latest.merge(fight_counts, on="fighter_id", how="left")
    latest["num_career_fights"] = latest["num_career_fights"].fillna(0).astype(int)
    return latest


def _latest(rows: pd.DataFrame) -> pd.Series:
    """Среди строк с дублирующимся именем — самый свежий бой."""
    return rows.sort_values("event_date").iloc[-1]


def find_fighter(index: pd.DataFrame, name_query: str) -> pd.Series:
    """Находит бойца по имени.

    Порядок разрешения:
      1. точное совпадение имени (без регистра) — берётся сразу;
      2. ровно одно (уникальное) совпадение по подстроке — берётся оно;
      3. иначе несколько кандидатов → ошибка со списком, чтобы не выбрать
         молча не того бойца.
    При дубликатах одного имени берётся боец с самым свежим боем.
    """
    q = name_query.strip().lower()
    names = index["fighter_name"].str.lower()

    exact = index[names == q]
    if len(exact):
        return _latest(exact)

    substr = index[names.str.contains(q, na=False, regex=False)]
    if len(substr) == 0:
        raise KeyError(f"Бойца '{name_query}' не нашёл")

    # уникальные имена среди совпадений
    unique_names = sorted(substr["fighter_name"].unique())
    if len(unique_names) == 1:
        return _latest(substr)

    shown = ", ".join(unique_names[:10])
    more = "" if len(unique_names) <= 10 else f" (и ещё {len(unique_names) - 10})"
    raise KeyError(
        f"Запрос '{name_query}' неоднозначен — подходят: {shown}{more}. "
        f"Уточните имя."
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


# aliases для случаев, где имя _diff не совпадает с базовой колонкой snapshot'а
DIFF_BASE_ALIAS = {
    "weight_cut": "weight_cut_lbs",
}


def _resolve_snap_value(snap: pd.Series, base: str) -> Any:
    if base in snap.index:
        return snap[base]
    alias = DIFF_BASE_ALIAS.get(base)
    if alias and alias in snap.index:
        return snap[alias]
    return np.nan


def build_pair_features(
    snap_a: pd.Series, snap_b: pd.Series, features: list[str], medians: dict[str, float]
) -> pd.DataFrame:
    """Собирает строку с 100 фичами из снапшотов двух бойцов.

    Антисимметричные _diff = a − b. Симметричные берутся из snapshot, либо
    вычисляются (layoff_*, weight_cut_*) либо падают на median.
    """
    row: dict[str, float] = {}
    layoff_a = snap_a.get("layoff_days", np.nan)
    layoff_b = snap_b.get("layoff_days", np.nan)
    cut_a = snap_a.get("weight_cut_lbs", np.nan)
    cut_b = snap_b.get("weight_cut_lbs", np.nan)

    for feat in features:
        val: float | None = None
        if feat.endswith("_diff"):
            base = feat[:-len("_diff")]
            va = _resolve_snap_value(snap_a, base)
            vb = _resolve_snap_value(snap_b, base)
            if not (pd.isna(va) or pd.isna(vb)):
                val = float(va) - float(vb)
            elif feat == "layoff_abs_diff" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = abs(float(layoff_a) - float(layoff_b))
        else:
            # симметричные spec-cases
            if feat == "layoff_max" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = float(max(layoff_a, layoff_b))
            elif feat == "layoff_min" and not (pd.isna(layoff_a) or pd.isna(layoff_b)):
                val = float(min(layoff_a, layoff_b))
            elif feat == "weight_cut_max" and not (pd.isna(cut_a) or pd.isna(cut_b)):
                val = float(max(cut_a, cut_b))
            elif feat == "weight_cut_sum" and not (pd.isna(cut_a) or pd.isna(cut_b)):
                val = float(cut_a + cut_b)
            elif feat == "stance_mismatch":
                sa = snap_a.get("southpaw", 0) or 0
                sb = snap_b.get("southpaw", 0) or 0
                val = int(sa != sb)
            elif feat == "same_stance":
                sa = str(snap_a.get("fighter_stance", "")).lower()
                sb = str(snap_b.get("fighter_stance", "")).lower()
                val = int(sa == sb and sa != "")
            elif feat == "style_clash":
                lean_a = snap_a.get("style_lean", np.nan)
                lean_b = snap_b.get("style_lean", np.nan)
                if not (pd.isna(lean_a) or pd.isna(lean_b)):
                    val = float(lean_a) * float(lean_b)
            else:
                v = snap_a.get(feat, np.nan)
                if not pd.isna(v):
                    val = float(v)

        if val is None:
            val = medians.get(feat, np.nan)
            val = float(val) if not pd.isna(val) else np.nan
        row[feat] = val

    return pd.DataFrame([row], columns=features)


def predict_p_win(snap_a: pd.Series, snap_b: pd.Series) -> dict[str, Any]:
    art = load_model()
    features = art["feature_columns"]
    medians = art["feature_medians"]
    model = art["model"]

    X = build_pair_features(snap_a, snap_b, features, medians)
    p = float(model.predict_proba(X)[0, 1])
    n_missing = int(X.isna().sum(axis=1).iloc[0])
    return {
        "p_a_wins": p,
        "n_features_missing": n_missing,
        "n_features_total": len(features),
    }
