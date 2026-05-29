"""Stage 6: rule-based fight plan generator (полированная версия).

CLI:
    python fight_plan.py "Ankalaev" "Pereira"
    python fight_plan.py "Khabib" "McGregor" --json

Объединяет:
  - P(win) из decay-HGB модели (calibrated)
  - стилевые архетипы из style_clusters
  - asymmetry diagnostics с Bayesian-shrinkage
  - envelope check (P10..P90 бойца)
  - percentile-based risk thresholds
И транслирует всё в русско-язычный текстовый план.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ufc_ablation_analysis import DATA_PATH, load_clean_fights
from asymmetry_diagnostics import (
    Advantage,
    compute_all_asymmetries,
    compute_ufc_globals,
    rank_asymmetries,
)
from envelope import build_envelope, fighter_envelope_summary
from plan_data import (
    ARTIFACT_DIR,
    CACHE_DIR,
    build_fighter_index,
    find_fighter,
    load_frame,
    predict_p_win,
)
from style_clustering import ARTIFACT_PATH as STYLE_ARTIFACT_PATH


# Используем CACHE_DIR (определён в plan_data — tempdir на read-only FS)
ENVELOPE_CACHE = CACHE_DIR / "_plan_envelope.joblib"
UFC_GLOBALS_CACHE = CACHE_DIR / "_plan_ufc_globals.joblib"


def _safe_joblib_dump(obj: Any, path) -> None:
    """Жалобно записывает в кеш; на read-only FS (cloud) — тихо игнорит."""
    try:
        joblib.dump(obj, path)
    except OSError:
        pass


def _load_envelope_cached(fights: pd.DataFrame):
    if ENVELOPE_CACHE.exists():
        return joblib.load(ENVELOPE_CACHE)
    per_fighter, global_env = build_envelope(fights)
    _safe_joblib_dump((per_fighter, global_env), ENVELOPE_CACHE)
    return per_fighter, global_env


def _load_ufc_globals_cached(idx: pd.DataFrame) -> dict[str, Any]:
    if UFC_GLOBALS_CACHE.exists():
        return joblib.load(UFC_GLOBALS_CACHE)
    g = compute_ufc_globals(idx)
    _safe_joblib_dump(g, UFC_GLOBALS_CACHE)
    return g


def _archetype_for(style_art: dict[str, Any], fighter_id: str) -> dict[str, Any]:
    cid = style_art["fighter_clusters"].get(fighter_id)
    if cid is None:
        return {"cluster_id": None, "name": "не определён", "hint": "нет данных"}
    return style_art["archetypes"][cid]


def _symmetric_matchup(
    style_art: dict[str, Any], cid_a: int, cid_b: int
) -> dict[str, Any]:
    n = np.array(style_art["matchup_n_fights"])
    w = np.array(style_art["matchup_f1_wins"])
    if cid_a == cid_b:
        n_total = int(n[cid_a, cid_b])
        return {
            "n_total": n_total,
            "a_win_rate": None if n_total == 0 else 0.5,
            "same_archetype": True,
        }
    n_ab = int(n[cid_a, cid_b])
    n_ba = int(n[cid_b, cid_a])
    w_ab = int(w[cid_a, cid_b])
    w_ba = int(w[cid_b, cid_a])
    a_wins_total = w_ab + (n_ba - w_ba)
    n_total = n_ab + n_ba
    if n_total == 0:
        return {"n_total": 0, "a_win_rate": None, "same_archetype": False}
    return {
        "n_total": n_total,
        "a_win_rate": a_wins_total / n_total,
        "same_archetype": False,
    }


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


def generate_plan(
    a_query: str, b_query: str, top_k: int = 3
) -> dict[str, Any]:
    warnings.filterwarnings("ignore")

    frame, features, diff_features, history, profile = load_frame()
    idx = build_fighter_index(frame, history, profile)
    ufc = _load_ufc_globals_cached(idx)

    a = find_fighter(idx, a_query)
    b = find_fighter(idx, b_query)
    fid_a = a["fighter_id"]
    fid_b = b["fighter_id"]

    pred = predict_p_win(a, b)

    style_art = joblib.load(STYLE_ARTIFACT_PATH)
    arch_a = _archetype_for(style_art, fid_a)
    arch_b = _archetype_for(style_art, fid_b)
    cid_a = arch_a["cluster_id"]
    cid_b = arch_b["cluster_id"]
    matchup_stats = (
        _symmetric_matchup(style_art, cid_a, cid_b)
        if (cid_a is not None and cid_b is not None)
        else {"n_total": 0, "a_win_rate": None, "same_archetype": False}
    )

    advs = rank_asymmetries(
        compute_all_asymmetries(a, b, ufc,
                                name_a=a["fighter_name"], name_b=b["fighter_name"])
    )
    exploit_a = [adv for adv in advs if adv.side == "A"]
    exploit_b = [adv for adv in advs if adv.side == "B"]

    fights = load_clean_fights(DATA_PATH)
    per_fighter_env, global_env = _load_envelope_cached(fights)
    env_a = fighter_envelope_summary(per_fighter_env, fid_a)
    env_b = fighter_envelope_summary(per_fighter_env, fid_b)

    return {
        "fighters": {
            "A": _fighter_card(a, arch_a),
            "B": _fighter_card(b, arch_b),
        },
        "verdict": {
            "p_a_wins": pred["p_a_wins"],
            "n_features_missing": pred["n_features_missing"],
            "n_features_total": pred["n_features_total"],
            "style_matchup": {
                "a_archetype": arch_a["name"],
                "b_archetype": arch_b["name"],
                "same_archetype": matchup_stats["same_archetype"],
                "historical_n": matchup_stats["n_total"],
                "historical_a_win_rate": matchup_stats["a_win_rate"],
            },
        },
        "exploit_for_A": [
            _dress_advantage(adv, env_a, env_b, "A", a["fighter_name"])
            for adv in exploit_a[:top_k]
        ],
        "exploit_for_B": [
            _dress_advantage(adv, env_a, env_b, "B", b["fighter_name"])
            for adv in exploit_b[:top_k]
        ],
        "pacing": _pacing_hints(a, b),
        "risk_warnings": _risk_warnings(a, b, ufc),
        "caveats": _default_caveats(pred),
    }


def _fighter_card(side: pd.Series, archetype: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": side["fighter_name"],
        "id": side["fighter_id"],
        "career_fights": int(side.get("num_career_fights", 0)),
        "age_years": _safe(side.get("age_years")),
        "reach_cm": _safe(side.get("reach_cm")),
        "height_cm": _safe(side.get("height_cm")),
        "stance": str(side.get("fighter_stance") or "?"),
        "archetype": archetype,
    }


def _safe(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


POS_SHARE_RU = {
    "head_share": "ударов в голову",
    "body_share": "ударов по корпусу",
    "leg_share": "лоу-киков",
    "distance_share": "ударов с дистанции",
    "clinch_share": "ударов в клинче",
    "ground_share": "ударов в партере",
}


def _dress_advantage(
    adv: Advantage, env_a: dict, env_b: dict, side: str, name: str
) -> dict[str, Any]:
    envelope_hint = None
    if adv.category == "strike_pos" and "_strikes_" in adv.name:
        pos = adv.name.split("_")[0]
        share_key = f"{pos}_share"
        env_side = env_a if side == "A" else env_b
        if share_key in env_side:
            e = env_side[share_key]
            envelope_hint = {
                "metric": share_key,
                "p10": round(e["p10"], 3),
                "p50": round(e["p50"], 3),
                "p90": round(e["p90"], 3),
                "n": int(e["n"]),
                "narrative": (
                    f"{name} исторически отдаёт {POS_SHARE_RU.get(share_key, share_key)} "
                    f"от {e['p10']*100:.0f}% до {e['p90']*100:.0f}% от своих ударов "
                    f"(в среднем {e['p50']*100:.0f}%, по {int(e['n'])} боям)"
                ),
            }
    return {
        "axis": adv.name,
        "category": adv.category,
        "magnitude": round(adv.magnitude, 3),
        "headline": adv.headline,
        "detail": adv.detail,
        "recommendation": adv.recommendation,
        "confidence": adv.confidence,
        "confidence_reason": adv.confidence_reason,
        "note": adv.note,
        "envelope": envelope_hint,
    }


def _pacing_hints(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    a_recent_def = _safe(a.get("ewma_striking_defense"))
    a_avg_def = _safe(a.get("avg_striking_defense"))
    b_recent_def = _safe(b.get("ewma_striking_defense"))
    b_avg_def = _safe(b.get("avg_striking_defense"))

    def trend(recent, career):
        if recent is None or career is None:
            return None
        return recent - career

    out = {
        "a_form_trend_defense": trend(a_recent_def, a_avg_def),
        "b_form_trend_defense": trend(b_recent_def, b_avg_def),
        "a_avg_fight_minutes": _safe(a.get("avg_fight_duration_min")),
        "b_avg_fight_minutes": _safe(b.get("avg_fight_duration_min")),
        "a_recent_volume": _safe(a.get("ewma_sig_landed_per_min")),
        "b_recent_volume": _safe(b.get("ewma_sig_landed_per_min")),
    }
    hints = []
    if out["b_form_trend_defense"] is not None and out["b_form_trend_defense"] < -0.03:
        hints.append(
            f"Защита B просела на {out['b_form_trend_defense']*100:.1f} п.п. от карьерной — "
            f"окно для давления (особенно во 2-3 раундах)"
        )
    if out["a_form_trend_defense"] is not None and out["a_form_trend_defense"] < -0.03:
        hints.append(
            f"Защита A просела на {out['a_form_trend_defense']*100:.1f} п.п. — "
            f"A бережно входить в обмены"
        )
    if (
        out["a_avg_fight_minutes"] is not None
        and out["b_avg_fight_minutes"] is not None
    ):
        delta = out["a_avg_fight_minutes"] - out["b_avg_fight_minutes"]
        if abs(delta) > 2.0:
            longer = "A" if delta > 0 else "B"
            hints.append(
                f"{longer} в среднем дольше держится в октагоне "
                f"({abs(delta):.1f} мин разница) — потенциальный cardio-перевес у {longer}"
            )
    out["hints"] = hints
    return out


def _risk_warnings(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any]
) -> list[str]:
    out = []
    p75_kd = ufc.get("p75_kd_per_15", 0.3)
    p75_td_att = ufc.get("p75_td_attempts_15", 4.0)
    p25_td_def = ufc.get("p25_td_defense", 0.55)
    p25_ko_loss_low = 0.08  # > 8% уже значимая склонность

    a_ko_loss = _safe(a.get("prior_ko_loss_rate"))
    b_ko_loss = _safe(b.get("prior_ko_loss_rate"))
    a_kd = _safe(a.get("avg_knockdowns_per_15"))
    b_kd = _safe(b.get("avg_knockdowns_per_15"))

    if a_ko_loss is not None and b_kd is not None:
        if a_ko_loss > p25_ko_loss_low and b_kd > p75_kd:
            out.append(
                f"⚠ A склонен к KO-проигрышам ({a_ko_loss:.0%}), а B бьёт нокдауны "
                f"{b_kd:.2f}/15мин (выше P75 UFC={p75_kd:.2f}) — A не идти на ранние размены"
            )
    if b_ko_loss is not None and a_kd is not None:
        if b_ko_loss > p25_ko_loss_low and a_kd > p75_kd:
            out.append(
                f"⚠ B склонен к KO-проигрышам ({b_ko_loss:.0%}), а A бьёт нокдауны "
                f"{a_kd:.2f}/15мин (выше P75 UFC={p75_kd:.2f}) — B уязвим к мощным ударам"
            )

    a_td_def = _safe(a.get("avg_takedown_defense"))
    b_td_vol = _safe(b.get("avg_takedowns_attempted_per_15"))
    if a_td_def is not None and b_td_vol is not None:
        if b_td_vol > p75_td_att and a_td_def < (1.0 - p25_td_def + 0.2):
            # т.е. TDD ниже UFC P75 (так как p25_td_def — нижний квартиль TDD)
            out.append(
                f"⚠ A's TDD {a_td_def:.0%} ниже элиты, B атакует {b_td_vol:.1f} TD/15мин "
                f"(выше P75 UFC={p75_td_att:.1f}) — A держать TDD активной работой ног"
            )
    b_td_def = _safe(b.get("avg_takedown_defense"))
    a_td_vol = _safe(a.get("avg_takedowns_attempted_per_15"))
    if b_td_def is not None and a_td_vol is not None:
        if a_td_vol > p75_td_att and b_td_def < (1.0 - p25_td_def + 0.2):
            out.append(
                f"⚠ B's TDD {b_td_def:.0%} ниже элиты, A атакует {a_td_vol:.1f} TD/15мин "
                f"(выше P75 UFC={p75_td_att:.1f}) — B избегать длительной борьбы у сетки"
            )

    return out


def _default_caveats(pred: dict[str, Any]) -> list[str]:
    out = [
        "Модель: HGB sigmoid-calibrated, обучена на decay-100 (см. UFC_DECAY_PIPELINE_FEATURES.md)",
        "Holdout ROC-AUC 0.68, log loss 0.64",
        "Рекомендации корреляционные, не каузальные — это анализ asymmetries, не RCT",
        "Архетипы — KMeans-кластеры (k=8) на cumulative tactical signature",
        "Accuracy позиционных ударов смягчена Bayesian-shrinkage'ом к UFC mean",
    ]
    if pred["n_features_missing"] > 5:
        out.append(
            f"⚠ {pred['n_features_missing']}/{pred['n_features_total']} фич упали на median imputation"
        )
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_plan(plan: dict[str, Any]) -> str:
    a = plan["fighters"]["A"]
    b = plan["fighters"]["B"]
    v = plan["verdict"]
    lines = []
    lines.append("═" * 75)
    lines.append(f"ПЛАН БОЯ: {a['name']}  vs  {b['name']}")
    lines.append("═" * 75)
    lines.append("")
    lines.append("ВЕРДИКТ")
    lines.append(
        f"  P({a['name']} побеждает) = {v['p_a_wins']:.3f}  "
        f"(missing features: {v['n_features_missing']}/{v['n_features_total']})"
    )
    sm = v["style_matchup"]
    lines.append(
        f"  Архетипы: A — «{sm['a_archetype']}», B — «{sm['b_archetype']}»"
    )
    if sm["same_archetype"]:
        lines.append(
            f"  ⓘ Оба бойца одного архетипа — стилевой край отсутствует, "
            f"исход определяется индивидуальной техникой и физикой"
        )
    elif sm["historical_a_win_rate"] is not None:
        wr = sm["historical_a_win_rate"]
        lines.append(
            f"  Историческая win rate {sm['a_archetype']} против {sm['b_archetype']}: "
            f"{wr:.1%} за A (n={sm['historical_n']} боёв, симметризованно)"
        )
    lines.append("")
    lines.append(
        f"  A: возраст {_age_str(a)}, размах {_reach_str(a)}, "
        f"стойка {a['stance']}, боёв в датасете {a['career_fights']}"
    )
    lines.append(f"     ⟶ {a['archetype']['hint']}")
    lines.append(
        f"  B: возраст {_age_str(b)}, размах {_reach_str(b)}, "
        f"стойка {b['stance']}, боёв в датасете {b['career_fights']}"
    )
    lines.append(f"     ⟶ {b['archetype']['hint']}")
    lines.append("")

    lines.append(f"ИСПОЛЬЗОВАТЬ — {a['name']} (топ-{len(plan['exploit_for_A'])})")
    if not plan["exploit_for_A"]:
        lines.append("  — нет явных асимметрий в пользу A")
    for i, adv in enumerate(plan["exploit_for_A"], 1):
        _render_adv_block(lines, i, adv)
    lines.append("")
    lines.append(f"ИЗБЕГАТЬ — то, что использует {b['name']}")
    if not plan["exploit_for_B"]:
        lines.append("  — нет явных асимметрий в пользу B")
    for i, adv in enumerate(plan["exploit_for_B"], 1):
        _render_adv_block(lines, i, adv)
    lines.append("")

    p = plan["pacing"]
    lines.append("ТЕМП И ФОРМА")
    if p["a_recent_volume"] is not None:
        lines.append(
            f"  A недавний объём ударов (EWMA): {p['a_recent_volume']:.2f}/мин, "
            f"средний бой ~{p['a_avg_fight_minutes']:.1f}мин"
        )
    if p["b_recent_volume"] is not None:
        lines.append(
            f"  B недавний объём ударов (EWMA): {p['b_recent_volume']:.2f}/мин, "
            f"средний бой ~{p['b_avg_fight_minutes']:.1f}мин"
        )
    for hint in p["hints"]:
        lines.append(f"  → {hint}")
    lines.append("")

    lines.append("РИСКИ")
    if not plan["risk_warnings"]:
        lines.append("  — нет крупных красных флагов")
    for w in plan["risk_warnings"]:
        lines.append(f"  {w}")
    lines.append("")

    lines.append("ОГОВОРКИ")
    for c in plan["caveats"]:
        lines.append(f"  • {c}")
    lines.append("═" * 75)
    return "\n".join(lines)


def _render_adv_block(lines: list[str], i: int, adv: dict[str, Any]) -> None:
    cat_icons = {
        "strike_pos": "🥊", "takedown": "🤼", "submission": "🐍",
        "ko": "💥", "stamina": "🫁", "physical": "📏",
    }
    icon = cat_icons.get(adv["category"], "•")
    lines.append(f"  {i}. {icon} {adv['headline']}")
    lines.append(f"     {adv['detail']}")
    lines.append(f"     → {adv['recommendation']}")
    conf_emoji = {"высокая": "✅", "средняя": "🟡", "низкая": "🟠",
                  "очень низкая": "🔴"}.get(adv["confidence"], "•")
    lines.append(
        f"     {conf_emoji} достоверность: {adv['confidence']} "
        f"({adv.get('confidence_reason', '')})"
    )
    if adv["envelope"] and adv["envelope"].get("narrative"):
        lines.append(f"     ⓘ {adv['envelope']['narrative']}")


def _age_str(side: dict[str, Any]) -> str:
    if side.get("age_years") is None:
        return "?"
    return f"{side['age_years']:.1f}"


def _reach_str(side: dict[str, Any]) -> str:
    if side.get("reach_cm") is None:
        return "?"
    return f"{side['reach_cm']:.0f}см"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сгенерировать план боя для пары бойцов UFC.")
    p.add_argument("a", type=str, help="Имя первого бойца (или подстрока)")
    p.add_argument("b", type=str, help="Имя второго бойца (или подстрока)")
    p.add_argument("--top-k", type=int, default=3, help="сколько exploit-осей на сторону")
    p.add_argument("--json", action="store_true", help="вывести JSON вместо текстового плана")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan = generate_plan(args.a, args.b, top_k=args.top_k)
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
    else:
        print(render_plan(plan))


if __name__ == "__main__":
    main()
