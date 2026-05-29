"""Stage 4 (v2): asymmetry diagnostics с человеко-читаемыми описаниями.

Каждая asymmetry даёт 4 поля:
  - headline:       краткий вывод одной фразой
  - detail:         расшифровка цифр обычным языком
  - recommendation: что делать на ринге
  - confidence:     "высокая" / "средняя" / "низкая" с обоснованием
И техническое поле note (для back-compat и debug-вывода).

Имена бойцов передаются в compute_all_asymmetries(a, b, ufc, name_a, name_b)
и вшиваются прямо в текст.

Magnitude гасится volume-weight, accuracy сглаживается Bayesian-shrinkage'ом
к UFC mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


POSITIONS = ["head", "body", "leg", "distance", "clinch", "ground"]
POS_RU = {
    "head": "голова",
    "body": "корпус",
    "leg": "ноги",
    "distance": "дистанция",
    "clinch": "клинч",
    "ground": "партер",
}
POS_RU_GENITIVE = {
    "head": "в голову",
    "body": "по корпусу",
    "leg": "по ногам",
    "distance": "на дистанции",
    "clinch": "в клинче",
    "ground": "в партере",
}


@dataclass
class Advantage:
    name: str
    category: str
    side: str                 # "A", "B", "neutral"
    magnitude: float
    a_value: float | None
    b_value: float | None
    # Новые поля для человеко-читаемого вывода:
    headline: str = ""
    detail: str = ""
    recommendation: str = ""
    confidence: str = ""       # "высокая"/"средняя"/"низкая"
    confidence_reason: str = ""
    # back-compat:
    note: str = ""
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


VOLUME_SCALE = {
    "head": 2.0,
    "body": 0.8,
    "leg": 0.6,
    "distance": 2.0,
    "clinch": 0.5,
    "ground": 0.5,
}

SHRINK_PRIOR_ATTEMPTS = 40.0


def _volume_weight(pos: str, volume: float) -> float:
    scale = VOLUME_SCALE.get(pos, 1.0)
    return min(volume, scale) / scale if scale > 0 else 0.0


def _estimated_total_attempts(side: pd.Series, pos: str) -> float:
    vol = _safe(side.get(f"avg_{pos}_attempted_per_min")) or 0.0
    dur = _safe(side.get("avg_fight_duration_min")) or 10.0
    n_fights = _safe(side.get("num_career_fights")) or 0.0
    return vol * dur * n_fights


def _shrink(raw: float, ufc_mean: float, n_attempts: float,
            prior: float = SHRINK_PRIOR_ATTEMPTS) -> float:
    if raw is None or ufc_mean is None:
        return raw
    if n_attempts <= 0:
        return ufc_mean
    w = n_attempts / (n_attempts + prior)
    return w * raw + (1.0 - w) * ufc_mean


def _confidence_from(volume_weight: float, n_attempts: float) -> tuple[str, str]:
    """Грубая оценка достоверности рекомендации."""
    if volume_weight >= 0.9 and n_attempts >= 80:
        return "высокая", f"объём бойца в этой зоне ≥{int(n_attempts)} попыток"
    if volume_weight >= 0.6 and n_attempts >= 30:
        return "средняя", f"объём ≈{int(n_attempts)} попыток, выборки достаточно"
    if volume_weight >= 0.3:
        return "низкая", f"редкая зона ({int(n_attempts)} попыток), accuracy шумна"
    return "очень низкая", "почти не работает в этой зоне — рекомендация на тонкой выборке"


# ---------------------------------------------------------------------------
# UFC globals
# ---------------------------------------------------------------------------


def compute_ufc_globals(fighter_index: pd.DataFrame, min_fights: int = 5) -> dict[str, Any]:
    pool = fighter_index[
        fighter_index["num_career_fights"].fillna(0) >= min_fights
    ].copy()
    g: dict[str, Any] = {"min_fights": min_fights, "pool_size": int(len(pool))}

    for pos in POSITIONS:
        for kind in ("accuracy", "defense"):
            col = f"avg_{pos}_{kind}"
            if col in pool.columns:
                g[f"mean_{pos}_{kind}"] = float(pool[col].mean(skipna=True))

    risk_metrics = {
        "avg_takedowns_attempted_per_15": "td_attempts_15",
        "avg_takedown_defense": "td_defense",
        "avg_knockdowns_per_15": "kd_per_15",
        "prior_ko_loss_rate": "ko_loss_rate",
        "avg_submissions_attempted_per_15": "sub_attempts_15",
    }
    for col, alias in risk_metrics.items():
        if col not in pool.columns:
            continue
        s = pool[col].dropna()
        if len(s) == 0:
            continue
        g[f"p25_{alias}"] = float(s.quantile(0.25))
        g[f"p50_{alias}"] = float(s.quantile(0.50))
        g[f"p75_{alias}"] = float(s.quantile(0.75))
        g[f"p90_{alias}"] = float(s.quantile(0.90))
    return g


# ---------------------------------------------------------------------------
# Per-position narrative templates
# ---------------------------------------------------------------------------

POSITION_RECOMMENDATION = {
    # Когда A атакует с перевесом — что A нужно делать
    "head": "Целить голову — бить с дистанции прямые и кроссы, искать KO-окно.",
    "body": "Грузить корпус: лоу-кросс, удары в печень — выматывать на длительность.",
    "leg": "Систематические лоу-кики — подсаживать ноги, ломать движение.",
    "distance": "Удерживать дистанцию, навязывать темп — соперник работает хуже на расстоянии.",
    "clinch": "Прижимать к сетке в клинче — соперник плохо защищается из захвата.",
    "ground": "Переводить бой в партер и работать ground-and-pound.",
}
POSITION_DEFEND_RECOMMENDATION = {
    # Когда B атакует с перевесом — что A нужно избегать
    "head": "Не открываться под удары в голову — держать защиту высокой, разрывать дистанцию после серий.",
    "body": "Закрывать корпус локтями — иначе размениваешь свой ресурс быстрее.",
    "leg": "Чек лоу-киков или отход — иначе ноги «сядут» к 3-му раунду.",
    "distance": "Сокращать дистанцию, не давать сопернику работать в его зоне.",
    "clinch": "Не дать прижать к сетке, разрывать клинч сразу.",
    "ground": "Не пускать в партер — sprawl + return to feet любой ценой.",
}


def _position_advantage_pair(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any], pos: str,
    name_a: str, name_b: str, attacker: str  # "A" или "B"
) -> Advantage | None:
    """Возвращает Advantage для одной из сторон-атакующих."""
    pos_ru = POS_RU[pos]
    pos_gen = POS_RU_GENITIVE[pos]

    if attacker == "A":
        att_side, def_side = a, b
        att_name, def_name = name_a, name_b
    else:
        att_side, def_side = b, a
        att_name, def_name = name_b, name_a

    acc_raw = _safe(att_side.get(f"avg_{pos}_accuracy"))
    def_raw = _safe(def_side.get(f"avg_{pos}_defense"))
    vol = _safe(att_side.get(f"avg_{pos}_attempted_per_min")) or 0.0
    if acc_raw is None or def_raw is None:
        return None

    att_total = _estimated_total_attempts(att_side, pos)
    # def-attempts: грубая оценка opp-volume в этой позиции по карьере защищающегося
    def_total = (
        (_safe(def_side.get("avg_fight_duration_min")) or 10.0)
        * (_safe(def_side.get("num_career_fights")) or 0.0)
        * max(
            _safe(att_side.get(f"avg_{pos}_attempted_per_min")) or 0.0,
            ufc.get(f"mean_{pos}_accuracy", 0.5) * 5,
        )
    )

    m_acc = ufc.get(f"mean_{pos}_accuracy")
    m_def = ufc.get(f"mean_{pos}_defense")
    acc = _shrink(acc_raw, m_acc, att_total) if m_acc is not None else acc_raw
    dfn = _shrink(def_raw, m_def, def_total) if m_def is not None else def_raw

    # acc = доля приземлённых попыток атакующего; dfn = доля отражённых
    # попыток защитника, т.е. защитник пропускает (1 - dfn). Перевес атаки —
    # это насколько точность атакующего превышает то, что защитник обычно
    # отдаёт. Совпадает по семантике с takedown-edge (td_acc - (1 - td_def)).
    edge = acc + dfn - 1.0
    w = _volume_weight(pos, vol)
    mag = abs(edge) * w

    # сторона, в чью пользу edge
    side = attacker if edge > 0 else ("B" if attacker == "A" else "A")

    # headline + detail
    if edge > 0:
        headline = (
            f"{pos_ru.capitalize()} — преимущество {att_name} в атаке: "
            f"+{int(round(edge * 100))} п.п. точности"
        )
        detail = (
            f"{att_name} приземляет {acc:.0%} попыток {pos_gen} "
            f"(сырой показатель {acc_raw:.0%}), а {def_name} пропускает "
            f"{1 - dfn:.0%} ({def_name} защищает {dfn:.0%} ударов). "
            f"Объём {att_name} в этой зоне — {vol:.2f} удара/мин."
        )
        recommendation = (
            f"{att_name}: {POSITION_RECOMMENDATION[pos]}"
        )
    else:
        headline = (
            f"{pos_ru.capitalize()} — {def_name} лучше защищается, чем {att_name} бьёт: "
            f"{abs(int(round(edge * 100)))} п.п. в пользу защиты"
        )
        detail = (
            f"{att_name} приземляет {acc:.0%} попыток {pos_gen}, но {def_name} "
            f"защищает на {dfn:.0%} (пропускает только {1 - dfn:.0%}). "
            f"Объём {att_name} здесь — {vol:.2f} удара/мин."
        )
        recommendation = (
            f"{att_name}: {POSITION_DEFEND_RECOMMENDATION[pos]}"
        )

    conf, conf_reason = _confidence_from(w, att_total)
    raw_shown = abs(acc_raw - acc) > 0.03  # покажем raw только если сильно отличается
    note = (
        f"{pos_ru.capitalize()}: точность {att_name} {acc:.0%}"
        + (f" (raw {acc_raw:.0%})" if raw_shown else "")
        + f" vs защита {def_name} {dfn:.0%} → перевес {edge:+.0%}, "
        f"объём {vol:.2f}/мин (w={w:.2f})"
    )

    return Advantage(
        name=f"{pos}_strikes_{attacker}_attack",
        category="strike_pos",
        side=side,
        magnitude=mag,
        a_value=acc if attacker == "A" else dfn,
        b_value=dfn if attacker == "A" else acc,
        headline=headline,
        detail=detail,
        recommendation=recommendation,
        confidence=conf,
        confidence_reason=conf_reason,
        note=note,
        components={
            "raw_attacker_acc": acc_raw,
            "shrunk_attacker_acc": acc,
            "raw_defender_def": def_raw,
            "shrunk_defender_def": dfn,
            "attacker_volume_per_min": vol,
            "volume_weight": w,
            "attacker_total_attempts_est": att_total,
        },
    )


def position_advantages(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    ufc = ufc or {}
    for pos in POSITIONS:
        for attacker in ("A", "B"):
            adv = _position_advantage_pair(a, b, ufc, pos, name_a, name_b, attacker)
            if adv is not None:
                out.append(adv)
    return out


# ---------------------------------------------------------------------------
# Takedown
# ---------------------------------------------------------------------------


def _takedown_pair(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any],
    name_a: str, name_b: str, attacker: str,
) -> Advantage | None:
    if attacker == "A":
        att_side, def_side = a, b
        att_name, def_name = name_a, name_b
    else:
        att_side, def_side = b, a
        att_name, def_name = name_b, name_a

    td_acc = _safe(att_side.get("avg_takedown_accuracy"))
    td_vol = _safe(att_side.get("avg_takedowns_attempted_per_15")) or 0.0
    td_def = _safe(def_side.get("avg_takedown_defense"))
    if td_acc is None or td_def is None:
        return None

    edge = td_acc - (1.0 - td_def)
    w = min(td_vol, 8.0) / 8.0
    mag = abs(edge) * w
    side = attacker if edge > 0 else ("B" if attacker == "A" else "A")

    p75_td = ufc.get("p75_td_attempts_15", 4.0)
    volume_descriptor = (
        "очень высокий" if td_vol > 8 else
        "высокий" if td_vol > p75_td else
        "средний" if td_vol > 2 else
        "низкий"
    )

    if edge > 0:
        headline = (
            f"Тейкдауны — преимущество {att_name}: точность атаки {td_acc:.0%} "
            f"против защиты {td_def:.0%}"
        )
        detail = (
            f"{att_name} приземляет {td_acc:.0%} попыток тейкдаунов, "
            f"а {def_name} защищается лишь на {td_def:.0%}. "
            f"Объём атак {att_name}: {td_vol:.1f} попыток за 15 минут ({volume_descriptor})."
        )
        recommendation = (
            f"{att_name}: давить chain-wrestling, особенно у сетки. "
            f"{def_name}: разрывать клинч, держать перемещение по центру."
        )
    else:
        headline = (
            f"Тейкдауны — {def_name} перекрывает атаку {att_name}: "
            f"TDD {td_def:.0%} vs точность {td_acc:.0%}"
        )
        detail = (
            f"{att_name} приземляет только {td_acc:.0%} попыток, а {def_name} "
            f"защищается на {td_def:.0%}. Темп атак {att_name}: {td_vol:.1f}/15мин."
        )
        recommendation = (
            f"{att_name}: переводы тут не зайдут — оставаться в стойке. "
            f"{def_name}: не отдавать сетку, продолжать TDD-работу."
        )

    if w >= 0.7:
        conf, conf_reason = "высокая", f"объём попыток {td_vol:.1f}/15мин — много данных"
    elif w >= 0.3:
        conf, conf_reason = "средняя", f"объём {td_vol:.1f}/15мин"
    else:
        conf, conf_reason = "низкая", f"редкие попытки ({td_vol:.1f}/15мин) — оценка шумная"

    return Advantage(
        name=f"takedown_{attacker}_attack",
        category="takedown",
        side=side,
        magnitude=mag,
        a_value=td_acc if attacker == "A" else td_def,
        b_value=td_def if attacker == "A" else td_acc,
        headline=headline,
        detail=detail,
        recommendation=recommendation,
        confidence=conf,
        confidence_reason=conf_reason,
        note=(
            f"Тейкдауны: точность {att_name} {td_acc:.0%} vs защита {def_name} "
            f"{td_def:.0%}, объём {td_vol:.1f}/15мин (w={w:.2f})"
        ),
        components={"td_acc": td_acc, "td_def": td_def,
                    "td_vol_per_15": td_vol, "volume_weight": w},
    )


def takedown_advantage(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    ufc = ufc or {}
    for attacker in ("A", "B"):
        adv = _takedown_pair(a, b, ufc, name_a, name_b, attacker)
        if adv is not None:
            out.append(adv)
    return out


# ---------------------------------------------------------------------------
# Submission threat
# ---------------------------------------------------------------------------


def submission_threat(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    ufc = ufc or {}
    p75 = ufc.get("p75_sub_attempts_15", 0.7)
    p90 = ufc.get("p90_sub_attempts_15", 1.5)

    for side_key, name, opp_name, side_obj in (
        ("A", name_a, name_b, a), ("B", name_b, name_a, b)
    ):
        sub_rate = _safe(side_obj.get("avg_submissions_attempted_per_15"))
        ctrl = _safe(side_obj.get("avg_control_minutes_per_15")) or 0.0
        if sub_rate is None:
            continue

        if sub_rate >= p90:
            tier = "элитный"
            danger_word = "острая"
            side = side_key
        elif sub_rate >= p75:
            tier = "выше среднего"
            danger_word = "значимая"
            side = side_key
        else:
            tier = "на уровне UFC"
            danger_word = "не главная"
            side = "neutral"

        headline = (
            f"Сабмишн-угроза от {name}: {tier} темп попыток "
            f"({sub_rate:.2f}/15мин)"
        )
        detail = (
            f"{name} пытается провести сабмишн {sub_rate:.2f} раз за 15 минут "
            f"(P75 по UFC = {p75:.2f}, P90 = {p90:.2f}). "
            f"Время контроля {name}: {ctrl:.1f} минут за 15."
        )
        if side != "neutral":
            recommendation = (
                f"{opp_name}: избегать партер и схватки на земле — это {danger_word} угроза. "
                f"Особенно осторожен у сетки и при попытках свипа."
            )
        else:
            recommendation = (
                f"{opp_name}: сабмишн — не главная опасность от {name}, но партер "
                f"всё равно лучше контролировать."
            )

        if sub_rate >= p90:
            conf, conf_reason = "высокая", "темп в топ-10% по UFC"
        elif sub_rate >= p75:
            conf, conf_reason = "средняя", "темп выше медианы UFC"
        else:
            conf, conf_reason = "низкая", "темп близок к средне-UFC"

        out.append(
            Advantage(
                name=f"submission_threat_{side_key}",
                category="submission",
                side=side,
                magnitude=sub_rate,
                a_value=sub_rate if side_key == "A" else None,
                b_value=sub_rate if side_key == "B" else None,
                headline=headline,
                detail=detail,
                recommendation=recommendation,
                confidence=conf,
                confidence_reason=conf_reason,
                note=f"Сабмишены {name}: {sub_rate:.2f}/15мин (P75 UFC={p75:.2f})",
                components={"sub_per_15": sub_rate, "ufc_p75": p75, "ufc_p90": p90,
                            "ctrl_per_15": ctrl},
            )
        )
    return out


# ---------------------------------------------------------------------------
# KO threat
# ---------------------------------------------------------------------------


def ko_advantage(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    ufc = ufc or {}
    p75_kd = ufc.get("p75_kd_per_15", 0.3)
    p90_kd = ufc.get("p90_kd_per_15", 0.8)

    for side_key, name, opp_name, side_obj, opp_obj in (
        ("A", name_a, name_b, a, b),
        ("B", name_b, name_a, b, a),
    ):
        kd = _safe(side_obj.get("avg_knockdowns_per_15"))
        opp_ko_loss = _safe(opp_obj.get("prior_ko_loss_rate"))
        if kd is None or opp_ko_loss is None:
            continue
        score = kd * (opp_ko_loss + 0.05)

        if kd >= p90_kd and opp_ko_loss >= 0.10:
            tier = "очень высокая"
            side = side_key
        elif kd >= p75_kd and opp_ko_loss >= 0.08:
            tier = "значимая"
            side = side_key
        elif kd >= p75_kd:
            tier = "средняя (соперник редко проигрывал KO)"
            side = "neutral"
        else:
            tier = "невысокая"
            side = "neutral"

        headline = (
            f"KO-угроза от {name}: {tier} "
            f"(KD {kd:.2f}/15мин, {opp_name} нокаутирован в {opp_ko_loss:.0%} прошлых боёв)"
        )
        detail = (
            f"{name} приземляет {kd:.2f} нокдауна за 15 минут (P75 UFC = {p75_kd:.2f}, "
            f"P90 = {p90_kd:.2f}). Доля прошлых поражений {opp_name} нокаутом — "
            f"{opp_ko_loss:.0%}."
        )
        if side != "neutral":
            recommendation = (
                f"{opp_name}: не идти на размен в первом раунде, "
                f"работать на разрыв и не открываться под выходные удары."
            )
        else:
            recommendation = (
                f"{opp_name}: KO — не приоритетная угроза, но головную защиту держать."
            )

        if kd >= p90_kd:
            conf, conf_reason = "высокая", "KD-rate в топ-10% UFC"
        elif kd >= p75_kd:
            conf, conf_reason = "средняя", "KD-rate выше P75 UFC"
        else:
            conf, conf_reason = "низкая", "KD-rate близок к среднему"

        out.append(
            Advantage(
                name=f"ko_threat_{side_key}",
                category="ko",
                side=side,
                magnitude=score,
                a_value=kd if side_key == "A" else opp_ko_loss,
                b_value=opp_ko_loss if side_key == "A" else kd,
                headline=headline,
                detail=detail,
                recommendation=recommendation,
                confidence=conf,
                confidence_reason=conf_reason,
                note=(
                    f"KO-угроза {name}: KD {kd:.2f}/15мин (P75 UFC={p75_kd:.2f}), "
                    f"{opp_name} проиграл нокаутом в {opp_ko_loss:.0%} боёв"
                ),
                components={"kd_per_15": kd, "opp_ko_loss_rate": opp_ko_loss,
                            "ufc_p75_kd": p75_kd, "ufc_p90_kd": p90_kd},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stamina / form trend
# ---------------------------------------------------------------------------


def stamina_advantage(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []

    for side_key, name, opp_name, side_obj in (
        ("A", name_a, name_b, a), ("B", name_b, name_a, b)
    ):
        recent_def = _safe(side_obj.get("ewma_striking_defense"))
        avg_def = _safe(side_obj.get("avg_striking_defense"))
        if recent_def is None or avg_def is None:
            continue
        delta = recent_def - avg_def

        if delta > 0.03:
            side = side_key
            trend_word = "выросла"
            headline = (
                f"Форма {name}: защита недавно {trend_word} "
                f"({recent_def:.0%} vs карьерные {avg_def:.0%}, "
                f"{delta:+.0%})"
            )
            recommendation = (
                f"{name} в хорошей форме — пик уверенности. "
                f"{opp_name}: не давать ему чувствовать темп, ломать ритм рваными атаками."
            )
        elif delta < -0.03:
            side = "B" if side_key == "A" else "A"  # форма соперника просела — выгодно нам
            trend_word = "просела"
            headline = (
                f"Форма {name}: защита недавно {trend_word} "
                f"({recent_def:.0%} vs карьерные {avg_def:.0%}, {delta:+.0%})"
            )
            recommendation = (
                f"{opp_name}: использовать просадку формы {name} — окно во 2-3 раундах "
                f"(давление на финишы, не давать ему «вкатиться» в бой)."
            )
        else:
            side = "neutral"
            trend_word = "стабильна"
            headline = (
                f"Форма {name}: стабильная защита ({recent_def:.0%} ≈ {avg_def:.0%})"
            )
            recommendation = f"{name} в обычной форме — без сюрпризов."

        detail = (
            f"Сравнение защиты {name} в последних боях (EWMA halflife=5) "
            f"vs карьерной средней. Recent: {recent_def:.0%}, average: {avg_def:.0%}. "
            f"Дельта {delta:+.0%}."
        )

        out.append(
            Advantage(
                name=f"form_trend_{side_key}_defense",
                category="stamina",
                side=side,
                magnitude=abs(delta),
                a_value=recent_def if side_key == "A" else avg_def,
                b_value=avg_def if side_key == "A" else recent_def,
                headline=headline,
                detail=detail,
                recommendation=recommendation,
                confidence="средняя",
                confidence_reason="EWMA с halflife=5 устойчив с 5+ боями в истории",
                note=(
                    f"Форма {name}: защита EWMA {recent_def:.0%} vs "
                    f"карьерная {avg_def:.0%} ({delta:+.0%})"
                ),
                components={"recent_def": recent_def, "career_def": avg_def, "delta": delta},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Physical
# ---------------------------------------------------------------------------


def physical_advantage(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    a_reach = _safe(a.get("reach_cm"))
    b_reach = _safe(b.get("reach_cm"))
    a_age = _safe(a.get("age_years"))
    b_age = _safe(b.get("age_years"))

    if a_reach is not None and b_reach is not None:
        delta = a_reach - b_reach
        side = "A" if delta > 1.5 else ("B" if delta < -1.5 else "neutral")
        mag = min(abs(delta) / 15.0, 1.0)
        longer_name = name_a if delta > 0 else name_b
        shorter_name = name_b if delta > 0 else name_a

        if side != "neutral":
            headline = (
                f"Размах рук — преимущество {longer_name}: +{abs(int(delta))} см"
            )
            detail = (
                f"{name_a}: {a_reach:.0f} см, {name_b}: {b_reach:.0f} см. "
                f"Разница {delta:+.0f} см в пользу {longer_name}."
            )
            recommendation = (
                f"{longer_name}: держать дистанцию, работать длинными ударами с переднего шага. "
                f"{shorter_name}: сокращать через клинч / удар «на отходе соперника»."
            )
        else:
            headline = f"Размах рук — паритет ({a_reach:.0f} vs {b_reach:.0f} см)"
            detail = f"Разница меньше 2 см — на бой это не влияет."
            recommendation = "Без рекомендаций — преимущества в размахе нет."

        out.append(
            Advantage(
                name="reach", category="physical", side=side, magnitude=mag,
                a_value=a_reach, b_value=b_reach,
                headline=headline, detail=detail, recommendation=recommendation,
                confidence="высокая", confidence_reason="антропометрия — без шума",
                note=f"Размах: {name_a} {a_reach:.0f}см vs {name_b} {b_reach:.0f}см ({delta:+.0f}см)",
                components={"a_reach": a_reach, "b_reach": b_reach},
            )
        )

    if a_age is not None and b_age is not None:
        delta = a_age - b_age
        if abs(delta) > 4:
            younger_name = name_b if delta > 0 else name_a
            older_name = name_a if delta > 0 else name_b
            side = "B" if delta > 4 else "A"
            headline = (
                f"Возраст — преимущество {younger_name}: моложе на "
                f"{abs(delta):.1f} лет"
            )
            detail = (
                f"{name_a}: {a_age:.1f} лет, {name_b}: {b_age:.1f}. "
                f"После 32-33 у бойцов обычно начинает падать скорость и кардио."
            )
            recommendation = (
                f"{younger_name}: тянуть бой в поздние раунды, выматывать темпом. "
                f"{older_name}: искать ранний финиш, не давать сопернику использовать молодость."
            )
            out.append(
                Advantage(
                    name="age_gap", category="physical", side=side,
                    magnitude=min(abs(delta) / 10.0, 1.0),
                    a_value=a_age, b_value=b_age,
                    headline=headline, detail=detail, recommendation=recommendation,
                    confidence="средняя", confidence_reason="возраст коррелирует с упадком, но индивидуально",
                    note=(
                        f"Возраст: {name_a} {a_age:.1f} vs {name_b} {b_age:.1f} "
                        f"({younger_name} моложе на {abs(delta):.1f} лет)"
                    ),
                    components={"a_age": a_age, "b_age": b_age},
                )
            )
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def compute_all_asymmetries(
    a: pd.Series, b: pd.Series, ufc: dict[str, Any] | None = None,
    name_a: str = "A", name_b: str = "B",
) -> list[Advantage]:
    out: list[Advantage] = []
    out.extend(position_advantages(a, b, ufc, name_a, name_b))
    out.extend(takedown_advantage(a, b, ufc, name_a, name_b))
    out.extend(submission_threat(a, b, ufc, name_a, name_b))
    out.extend(ko_advantage(a, b, ufc, name_a, name_b))
    out.extend(stamina_advantage(a, b, ufc, name_a, name_b))
    out.extend(physical_advantage(a, b, ufc, name_a, name_b))
    return out


# Шкалы приводят сырой magnitude каждой категории к ~[0,1]: значение ≈ 1.0
# соответствует «явно значимому» перевесу в этой категории. Без этого top-K
# доминировался бы категориями с крупной числовой шкалой (submission ~ сырой
# темп до 2.0, reach ~ до 1.0), а KO/форма (≤0.1) никогда не всплывали.
CATEGORY_MAGNITUDE_SCALE = {
    "strike_pos": 0.20,   # |acc+dfn-1|·w; сильный перевес ~0.15
    "takedown": 0.30,     # |td_acc-(1-td_def)|·w
    "submission": 1.20,   # сырой sub_rate/15мин; P90 ~1.5
    "ko": 0.15,           # kd·(opp_ko_loss+0.05)
    "stamina": 0.08,      # |Δ defense|; порог значимости 0.03
    "physical": 1.00,     # reach/age уже нормированы в [0,1]
}

# Достоверность входит множителем, чтобы шумная ось не обгоняла надёжную.
CONFIDENCE_WEIGHT = {
    "высокая": 1.0,
    "средняя": 0.8,
    "низкая": 0.55,
    "очень низкая": 0.3,
    "": 0.7,
}


def _rank_priority(adv: Advantage) -> float:
    """Сопоставимый между категориями приоритет ∈ ~[0, 1.5]."""
    scale = CATEGORY_MAGNITUDE_SCALE.get(adv.category, 1.0)
    norm = adv.magnitude / scale if scale > 0 else 0.0
    norm = min(norm, 1.5)  # клампим выброс, чтобы один экстрим не доминировал
    conf_w = CONFIDENCE_WEIGHT.get(adv.confidence, 0.7)
    return norm * conf_w


def rank_asymmetries(advs: list[Advantage]) -> list[Advantage]:
    """Ранжирует по нормированному кросс-категорийному приоритету (не по сырому
    magnitude — у категорий разные шкалы, см. CATEGORY_MAGNITUDE_SCALE)."""
    return sorted(advs, key=_rank_priority, reverse=True)


def split_by_side(
    advs: list[Advantage],
) -> tuple[list[Advantage], list[Advantage], list[Advantage]]:
    a, b, n = [], [], []
    for adv in advs:
        if adv.side == "A":
            a.append(adv)
        elif adv.side == "B":
            b.append(adv)
        else:
            n.append(adv)
    return a, b, n


def to_records(advs: list[Advantage]) -> list[dict[str, Any]]:
    return [asdict(adv) for adv in advs]
