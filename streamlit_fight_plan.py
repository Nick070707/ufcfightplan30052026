"""Streamlit demo: UFC Fight Plan Generator.

Запуск локально:
    streamlit run streamlit_fight_plan.py

Облако Streamlit:
  - root репозитория должен содержать этот файл
  - requirements.txt — со списком пакетов
  - UFC_full_data_silver.csv в корне
  - artifacts/ufc_decay_hist_gradient_boosting_sigmoid_calibrated.joblib
  - artifacts/style_clusters.joblib (генерируется style_clustering.py)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from asymmetry_diagnostics import (
    compute_all_asymmetries,
    compute_ufc_globals,
    rank_asymmetries,
)
from envelope import build_envelope, fighter_envelope_summary
from plan_data import (
    build_fighter_index,
    build_pair_features,
    find_fighter,
    load_frame,
    load_model,
)
from style_clustering import ARTIFACT_PATH as STYLE_ARTIFACT_PATH
from ufc_ablation_analysis import DATA_PATH, load_clean_fights


warnings.filterwarnings("ignore")



st.set_page_config(page_title="UFC Fight Plan", layout="wide", page_icon="")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Загружаю decay-100 frame и snapshot бойцов…")
def get_frame_bundle():
    frame, features, diff_features, history, profile = load_frame()
    idx = build_fighter_index(frame, history, profile)
    return frame, features, diff_features, idx


@st.cache_resource(show_spinner="Загружаю калиброванную HGB модель…")
def get_model_artifact() -> dict[str, Any]:
    return load_model()


@st.cache_resource(show_spinner="Загружаю стилевые кластеры…")
def get_style_artifact() -> dict[str, Any]:
    if not Path(STYLE_ARTIFACT_PATH).exists():
        st.error(
            "Не найден style_clusters.joblib. Запусти: "
            "`python style_clustering.py --k 8`"
        )
        st.stop()
    return joblib.load(STYLE_ARTIFACT_PATH)


@st.cache_resource(show_spinner="Считаю UFC global percentiles…")
def get_ufc_globals(_idx: pd.DataFrame) -> dict[str, Any]:
    return compute_ufc_globals(_idx)


@st.cache_resource(show_spinner="Строю envelope на основе per-fight rates…")
def get_envelope():
    fights = load_clean_fights(DATA_PATH)
    per_fighter, global_env = build_envelope(fights)
    return per_fighter, global_env


# ---------------------------------------------------------------------------
# Plan computation (reuse fight_plan.generate_plan но без диск-кеша)
# ---------------------------------------------------------------------------


def _archetype_for(style_art: dict[str, Any], fighter_id: str) -> dict[str, Any]:
    cid = style_art["fighter_clusters"].get(fighter_id)
    if cid is None:
        return {"cluster_id": None, "name": "не определён", "hint": "нет данных"}
    return style_art["archetypes"][cid]


def _symmetric_matchup(style_art: dict[str, Any], cid_a: int, cid_b: int) -> dict[str, Any]:
    n = np.array(style_art["matchup_n_fights"])
    w = np.array(style_art["matchup_f1_wins"])
    if cid_a == cid_b:
        n_total = int(n[cid_a, cid_b])
        return {"n_total": n_total, "a_win_rate": None if n_total == 0 else 0.5,
                "same_archetype": True}
    n_ab, n_ba = int(n[cid_a, cid_b]), int(n[cid_b, cid_a])
    w_ab, w_ba = int(w[cid_a, cid_b]), int(w[cid_b, cid_a])
    a_wins_total = w_ab + (n_ba - w_ba)
    n_total = n_ab + n_ba
    if n_total == 0:
        return {"n_total": 0, "a_win_rate": None, "same_archetype": False}
    return {"n_total": n_total, "a_win_rate": a_wins_total / n_total,
            "same_archetype": False}


def predict_p_win_from_snaps(a: pd.Series, b: pd.Series, art: dict[str, Any]) -> dict[str, Any]:
    X = build_pair_features(a, b, art["feature_columns"], art["feature_medians"])
    p = float(art["model"].predict_proba(X)[0, 1])
    n_missing = int(X.isna().sum(axis=1).iloc[0])
    return {"p_a_wins": p, "n_features_missing": n_missing,
            "n_features_total": len(art["feature_columns"])}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar(idx: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Параметры")
    min_fights = st.sidebar.slider(
        "Минимум боёв в датасете", 0, 25, value=5,
        help="Бойцы с < N боёв имеют шумные cumulative-метрики"
    )
    top_k = st.sidebar.slider("Сколько exploit-осей показывать", 1, 8, value=4)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Алгоритм: P(win) от HGB-калиброванной модели decay-100 + "
        "ранжированные asymmetries (Bayesian-shrinkage позиционной accuracy) + "
        "стилевые архетипы (KMeans k=8) + envelope check (P10/P50/P90 бойца)."
    )
    st.sidebar.caption(
        "Подробнее: см. **UFC_FIGHT_PLAN_ALGORITHM.md**."
    )
    return {"min_fights": min_fights, "top_k": top_k}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def fighter_picker(
    label: str, options: list[str], default_query: str, key: str
) -> str:
    default_idx = 0
    for i, opt in enumerate(options):
        if default_query.lower() in opt.lower():
            default_idx = i
            break
    return st.selectbox(label, options, index=default_idx, key=key)


def render_archetype_card(side: str, name: str, archetype: dict[str, Any], snap: pd.Series):
    st.markdown(f"### {side}: {name}")
    age = snap.get("age_years")
    reach = snap.get("reach_cm")
    stance = snap.get("fighter_stance") or "?"
    fights = int(snap.get("num_career_fights", 0))
    st.caption(
        f"Возраст: {age:.1f} • Размах: {reach:.0f}см • Стойка: {stance} • Боёв в датасете: {fights}"
        if age is not None and reach is not None else f"Боёв в датасете: {fights}"
    )
    st.info(f"**Архетип:** {archetype['name']}  \n_{archetype['hint']}_")


def render_advantage(adv: dict[str, Any], idx: int):
    cat_icon = {
        "strike_pos": "🥊", "takedown": "🤼", "submission": "🐍",
        "ko": "💥", "stamina": "🫁", "physical": "📏",
    }.get(adv["category"], "•")
    conf_emoji = {"высокая": "✅", "средняя": "🟡",
                  "низкая": "🟠", "очень низкая": "🔴"}.get(adv.get("confidence", ""), "•")

    with st.container(border=True):
        # Headline крупно
        st.markdown(f"**{idx}. {cat_icon} {adv.get('headline') or adv.get('note', '')}**")
        # Detail обычным текстом
        if adv.get("detail"):
            st.markdown(adv["detail"])
        # Recommendation выделенно
        if adv.get("recommendation"):
            st.markdown(f"**→ {adv['recommendation']}**")
        # Confidence + envelope в одну подпись
        meta_parts = []
        if adv.get("confidence"):
            meta_parts.append(
                f"{conf_emoji} **достоверность:** {adv['confidence']}"
                + (f" ({adv.get('confidence_reason', '')})" if adv.get('confidence_reason') else "")
            )
        if adv.get("envelope") and adv["envelope"].get("narrative"):
            meta_parts.append(f"ⓘ {adv['envelope']['narrative']}")
        if meta_parts:
            st.caption("  \n".join(meta_parts))
        # Магнитуда в одну строку под капотом (для тех, кто понимает)
        with st.expander("технические детали", expanded=False):
            st.text(adv.get("note", ""))
            st.metric("Магнитуда", f"{adv['magnitude']:.3f}")


def main() -> None:
    st.title("🥋 UFC Fight Plan Generator")
    st.markdown(
        "Генерация плана боя для произвольной пары бойцов UFC. "
        "Вероятность победы — calibrated HGB на 100 фичах (decay-100). "
        "Рекомендации — ранжированные асимметрии, переведённые в текстовый план."
    )

    frame, features, diff_features, idx = get_frame_bundle()
    art = get_model_artifact()
    style_art = get_style_artifact()
    ufc = get_ufc_globals(idx)
    per_fighter_env, _ = get_envelope()

    cfg = sidebar(idx)

    selectable = idx[idx["num_career_fights"].fillna(0) >= cfg["min_fights"]].copy()
    selectable = selectable.sort_values("fighter_name").reset_index(drop=True)
    names = selectable["fighter_name"].tolist()

    if len(names) < 2:
        st.warning("Слишком строгий фильтр по числу боёв. Уменьши в сайдбаре.")
        st.stop()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        name_a = fighter_picker("Боец A (углом A)", names, "Khabib", "pick_a")
    with col2:
        name_b = fighter_picker("Боец B (углом B)", names, "McGregor", "pick_b")

    if name_a == name_b:
        st.warning("Выбери двух разных бойцов.")
        st.stop()

    a = selectable[selectable["fighter_name"] == name_a].iloc[0]
    b = selectable[selectable["fighter_name"] == name_b].iloc[0]
    fid_a, fid_b = a["fighter_id"], b["fighter_id"]

    # === ВЕРДИКТ ===
    pred = predict_p_win_from_snaps(a, b, art)
    arch_a = _archetype_for(style_art, fid_a)
    arch_b = _archetype_for(style_art, fid_b)
    matchup = _symmetric_matchup(style_art, arch_a["cluster_id"], arch_b["cluster_id"])

    st.markdown("## 📊 Вердикт")
    pa = pred["p_a_wins"]
    pb = 1.0 - pa
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"P({name_a} побеждает)", f"{pa:.1%}")
        st.progress(pa)
    with c2:
        st.metric(f"P({name_b} побеждает)", f"{pb:.1%}")
        st.progress(pb)

    if pred["n_features_missing"] > 0:
        st.caption(
            f"⚠ {pred['n_features_missing']}/{pred['n_features_total']} фич упали на median imputation"
        )

    if matchup["same_archetype"]:
        st.info(
            f"ⓘ Оба бойца одного архетипа («{arch_a['name']}») — стилевой край отсутствует, "
            f"исход определяется индивидуальной техникой и физикой."
        )
    elif matchup["a_win_rate"] is not None:
        wr = matchup["a_win_rate"]
        st.info(
            f"Историческая win rate **{arch_a['name']}** vs **{arch_b['name']}**: "
            f"**{wr:.1%}** за A (n={matchup['n_total']} боёв, симметризованно по перестановке углов)."
        )

    # === КАРТОЧКИ ===
    st.markdown("## 👤 Карточки бойцов")
    c1, c2 = st.columns(2)
    with c1:
        render_archetype_card("A", name_a, arch_a, a)
    with c2:
        render_archetype_card("B", name_b, arch_b, b)

    # === ASYMMETRIES ===
    advs = rank_asymmetries(
        compute_all_asymmetries(a, b, ufc, name_a=name_a, name_b=name_b)
    )
    exploit_a = [adv for adv in advs if adv.side == "A"][: cfg["top_k"]]
    exploit_b = [adv for adv in advs if adv.side == "B"][: cfg["top_k"]]
    env_a = fighter_envelope_summary(per_fighter_env, fid_a)
    env_b = fighter_envelope_summary(per_fighter_env, fid_b)

    pos_share_ru = {
        "head_share": "ударов в голову", "body_share": "ударов по корпусу",
        "leg_share": "лоу-киков", "distance_share": "ударов с дистанции",
        "clinch_share": "ударов в клинче", "ground_share": "ударов в партере",
    }

    def _dress(adv, env_self, who_name):
        env_hint = None
        if adv.category == "strike_pos" and "_strikes_" in adv.name:
            pos = adv.name.split("_")[0]
            sk = f"{pos}_share"
            if sk in env_self:
                e = env_self[sk]
                env_hint = {
                    "metric": sk,
                    "p10": round(e["p10"], 3),
                    "p50": round(e["p50"], 3),
                    "p90": round(e["p90"], 3),
                    "n": int(e["n"]),
                    "narrative": (
                        f"{who_name} исторически отдаёт {pos_share_ru.get(sk, sk)} "
                        f"от {e['p10']*100:.0f}% до {e['p90']*100:.0f}% от своих ударов "
                        f"(в среднем {e['p50']*100:.0f}%, по {int(e['n'])} боям)"
                    ),
                }
        return {
            "axis": adv.name, "category": adv.category,
            "magnitude": round(adv.magnitude, 3),
            "headline": adv.headline, "detail": adv.detail,
            "recommendation": adv.recommendation,
            "confidence": adv.confidence, "confidence_reason": adv.confidence_reason,
            "note": adv.note, "envelope": env_hint,
        }

    st.markdown("## ⚔ Эксплоиты")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### Использовать — {name_a}")
        if not exploit_a:
            st.write("— нет явных асимметрий в пользу A")
        for i, adv in enumerate(exploit_a, 1):
            render_advantage(_dress(adv, env_a, name_a), i)
    with c2:
        st.markdown(f"### Избегать — то, что использует {name_b}")
        if not exploit_b:
            st.write("— нет явных асимметрий в пользу B")
        for i, adv in enumerate(exploit_b, 1):
            render_advantage(_dress(adv, env_b, name_b), i)

    # === PACING & RISKS ===
    from fight_plan import _pacing_hints, _risk_warnings
    pacing = _pacing_hints(a, b)
    risks = _risk_warnings(a, b, ufc)

    st.markdown("## 🫁 Темп и форма")
    pc1, pc2 = st.columns(2)
    with pc1:
        if pacing["a_recent_volume"] is not None:
            st.metric(
                f"{name_a} — недавний sig-volume (EWMA)",
                f"{pacing['a_recent_volume']:.2f}/мин",
                delta=f"средний бой ~{pacing['a_avg_fight_minutes']:.1f} мин"
                      if pacing['a_avg_fight_minutes'] else None,
            )
    with pc2:
        if pacing["b_recent_volume"] is not None:
            st.metric(
                f"{name_b} — недавний sig-volume (EWMA)",
                f"{pacing['b_recent_volume']:.2f}/мин",
                delta=f"средний бой ~{pacing['b_avg_fight_minutes']:.1f} мин"
                      if pacing['b_avg_fight_minutes'] else None,
            )
    for hint in pacing["hints"]:
        st.write(f"→ {hint}")

    st.markdown("## ⚠ Риски")
    if not risks:
        st.write("— нет крупных красных флагов")
    for r in risks:
        st.warning(r)

    # === RAW DATA EXPANDERS ===
    with st.expander("📈 Все asymmetries (включая neutral)"):
        records = [
            {
                "сторона": adv.side, "категория": adv.category,
                "ось": adv.name, "магнитуда": round(adv.magnitude, 3),
                "описание": adv.note,
            } for adv in advs
        ]
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")

    with st.expander("📋 Полная карточка снапшота — A"):
        st.dataframe(_snapshot_view(a), hide_index=True, width="stretch")
    with st.expander("📋 Полная карточка снапшота — B"):
        st.dataframe(_snapshot_view(b), hide_index=True, width="stretch")

    with st.expander("🧠 Метаданные модели"):
        st.json({
            **art["metadata"],
            "p_a_wins": pa, "n_features_missing": pred["n_features_missing"],
        })

    with st.expander("ℹ Алгоритм"):
        try:
            md = Path("UFC_FIGHT_PLAN_ALGORITHM.md").read_text(encoding="utf-8")
            st.markdown(md)
        except FileNotFoundError:
            st.write("UFC_FIGHT_PLAN_ALGORITHM.md не найден в корне.")


def _snapshot_view(snap: pd.Series) -> pd.DataFrame:
    interesting = [
        "num_career_fights", "age_years", "height_cm", "reach_cm",
        "fighter_stance", "fighter_weight_lbs",
        "avg_sig_strikes_landed_per_min", "avg_sig_strikes_attempted_per_min",
        "avg_sig_strikes_absorbed_per_min", "avg_striking_accuracy", "avg_striking_defense",
        "avg_takedowns_attempted_per_15", "avg_takedown_accuracy", "avg_takedown_defense",
        "avg_submissions_attempted_per_15", "avg_control_minutes_per_15",
        "avg_knockdowns_per_15", "avg_knockdowns_absorbed_per_15",
        "avg_head_strike_share", "avg_body_strike_share", "avg_leg_strike_share",
        "avg_distance_strike_share", "avg_clinch_strike_share", "avg_ground_strike_share",
        "prior_win_rate", "prior_ko_loss_rate",
        "ewma_sig_landed_per_min", "ewma_striking_defense",
        "ewma_takedowns_landed_per_15", "ewma_takedown_defense", "ewma_knockdowns_per_15",
    ]
    rows = []
    for col in interesting:
        if col in snap.index:
            val = snap[col]
            if pd.isna(val):
                continue
            rows.append({"метрика": col, "значение": val})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
