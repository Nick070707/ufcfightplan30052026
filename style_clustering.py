"""Stage 3: style clustering + cluster × cluster matchup matrix.

Кластеризует бойцов по their cumulative tactical signature на момент
последнего боя. Эвристика именования кластеров по центроидам. Считает
матрицу историческихwin rates cluster_A × cluster_B (где B — соперник).

Запуск:
    python style_clustering.py
    python style_clustering.py --k 7
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# UTF-8 stdout, cp1252 в Windows shell режет кириллицу
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from plan_data import ARTIFACT_DIR, build_fighter_index, load_frame
from style_transform import HEAVY_TAIL_FEATURES, SIGNATURE_FEATURES, log1p_heavy


ARTIFACT_PATH = ARTIFACT_DIR / "style_clusters.joblib"
SUMMARY_PATH = ARTIFACT_DIR / "style_clusters_summary.json"

# SIGNATURE_FEATURES и HEAVY_TAIL_FEATURES / log1p_heavy — в style_transform
# (стабильный __module__ для пикла FunctionTransformer). Реэкспорт для
# обратной совместимости импортов.

RANDOM_STATE = 42
MIN_FIGHTS_FOR_CLUSTERING = 5


def fit_clusters(
    signature: pd.DataFrame, k: int, random_state: int = RANDOM_STATE
) -> tuple[Pipeline, np.ndarray]:
    pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(log1p_heavy, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=k, n_init=20, random_state=random_state),
            ),
        ]
    )
    labels = pipe.fit_predict(signature.values)
    return pipe, labels


def raw_centroids(
    signature: pd.DataFrame, labels: np.ndarray, k: int
) -> pd.DataFrame:
    """Центроиды в ИСХОДНОЙ (raw) шкале — средние фич по членам кластера.

    Считаем по оригинальным (не трансформированным) значениям, чтобы именование
    и summary были интерпретируемы независимо от log1p-преобразования внутри
    KMeans-пайплайна.
    """
    df = signature.copy()
    df["__cluster"] = labels
    cent = df.groupby("__cluster")[SIGNATURE_FEATURES].mean()
    return cent.reindex(range(k))


# Архетипы как vector signature над z-нормализованными центроидами.
# Каждый — взвешенная сумма z-score по фичам. Лучший score → имя кластера.
# Положительный вес = фича выше среднего; отрицательный — ниже.
ARCHETYPE_SIGNATURES: list[dict[str, Any]] = [
    {
        "name": "борец-контролёр",
        "hint": "много тейкдаунов и контроля, давление в партере",
        "weights": {
            "avg_takedowns_attempted_per_15": 1.2,
            "avg_control_minutes_per_15": 1.4,
            "avg_takedown_defense": 1.0,
            "avg_ground_strike_share": 0.8,
            "avg_takedown_accuracy": 0.5,
            "avg_distance_strike_share": -0.7,
        },
    },
    {
        "name": "сабмишн-специалист",
        "hint": "много попыток сабмишена, работа в партере",
        "weights": {
            "avg_submissions_attempted_per_15": 1.8,
            "avg_control_minutes_per_15": 0.6,
            "avg_ground_strike_share": 0.6,
            "avg_takedowns_attempted_per_15": 0.4,
        },
    },
    {
        "name": "нокаутёр",
        "hint": "высокий KD rate на дистанции, заканчивает рано",
        "weights": {
            "avg_knockdowns_per_15": 1.8,
            "avg_distance_strike_share": 0.6,
            "avg_fight_duration_min": -0.8,
            "avg_striking_accuracy": 0.4,
        },
    },
    {
        "name": "темповик-штурмовик",
        "hint": "высокий темп ударов на дистанции, идёт в обмены",
        "weights": {
            # NB: avg_sig_strikes_attempted_per_min нет в SIGNATURE_FEATURES —
            # раньше висел мёртвым весом (z=0) и занижал score шаблона. Заменён
            # на sig_absorbed (штурмовик идёт в размены и сам пропускает).
            "avg_sig_strikes_landed_per_min": 1.2,
            "avg_sig_strikes_absorbed_per_min": 0.6,
            "avg_distance_strike_share": 0.7,
            "avg_head_strike_share": 0.4,
            "avg_takedowns_attempted_per_15": -0.5,
        },
    },
    {
        "name": "защитный аутфайтер",
        "hint": "высокая защита, мало пропускает, контролирует дистанцию",
        "weights": {
            "avg_striking_defense": 1.4,
            "avg_sig_strikes_absorbed_per_min": -1.2,
            "avg_distance_strike_share": 0.6,
            "avg_fight_duration_min": 0.4,
            "avg_takedowns_attempted_per_15": -0.8,  # pure striker, не борец
            "avg_control_minutes_per_15": -0.6,
        },
    },
    {
        "name": "лоу-кикер",
        "hint": "акцент на удары по ногам",
        "weights": {
            "avg_leg_strike_share": 1.8,
            "avg_head_strike_share": -0.8,
            "avg_clinch_strike_share": -0.4,  # чистый кикер, не клинчер
        },
    },
    {
        "name": "ближний боец",
        "hint": "много клинча и работы по корпусу, давит вплотную",
        "weights": {
            "avg_clinch_strike_share": 1.5,
            "avg_body_strike_share": 1.0,
            "avg_distance_strike_share": -0.8,
            "avg_head_strike_share": -0.6,
        },
    },
    {
        "name": "терпеливый решатель",
        "hint": "тянет в решение, низкий темп",
        "weights": {
            "avg_fight_duration_min": 1.4,
            "avg_sig_strikes_landed_per_min": -0.8,
            "avg_knockdowns_per_15": -0.5,
            "avg_striking_accuracy": -0.4,
        },
    },
    {
        "name": "разносторонний грэпплер",
        "hint": "комбинирует борьбу и партер без явной доминанты",
        "weights": {
            "avg_takedowns_attempted_per_15": 0.8,
            "avg_submissions_attempted_per_15": 0.5,
            "avg_control_minutes_per_15": 0.5,
            "avg_ground_strike_share": 0.4,
        },
    },
    {
        "name": "техничный универсал",
        "hint": "высокая защита, длинные бои, разносторонняя техника без явной доминанты",
        # Для кластера техничных «дистанционеров-универсалов» (Jones/GSP/Usman):
        # высокая защита + длинные бои + разнообразие ударов (в т.ч. по ногам),
        # но без перекоса в чистый лоу-кик. Перебивает «лоу-кикера» на таком
        # кластере, оставляя ему лишь по-настоящему ногобойные кластеры.
        "weights": {
            "avg_striking_defense": 1.5,
            "avg_fight_duration_min": 1.0,
            "avg_leg_strike_share": 1.0,
        },
    },
]

MIN_SIGNATURE_SCORE = 0.6  # ниже — кластер «смешанный»


def _score_archetype(zrow: pd.Series, weights: dict[str, float]) -> float:
    """Взвешенная сумма z-score по указанным фичам, нормированная на |w|-sum."""
    total = 0.0
    norm = 0.0
    for feat, w in weights.items():
        z = float(zrow.get(feat, 0.0))
        total += w * z
        norm += abs(w)
    return total / norm if norm > 0 else 0.0


def name_clusters(centroids: pd.DataFrame) -> list[dict[str, Any]]:
    """Жадно-уникальная привязка архетипов к кластерам.

    Раньше каждый кластер независимо брал свой argmax-шаблон и при коллизии
    получал суффикс «-2» (а похожие кластеры дублировали имя). Теперь решаем
    как задачу назначения: из всех пар (кластер, шаблон) по убыванию score
    жадно фиксируем пару, если и кластер, и имя ещё свободны. Так каждый
    кластер получает СВОЙ лучший ещё не занятый архетип — без дублей и без «-2».
    """
    means = centroids.mean()
    stds = centroids.std().replace(0, 1)
    zmat = ((centroids - means) / stds).fillna(0.0)

    # score(cluster, template) для всех пар
    triples: list[tuple[float, int, dict[str, Any]]] = []
    for cid in centroids.index:
        zrow = zmat.loc[cid]
        for sig in ARCHETYPE_SIGNATURES:
            triples.append((_score_archetype(zrow, sig["weights"]), int(cid), sig))
    triples.sort(key=lambda t: t[0], reverse=True)

    assigned: dict[int, tuple[dict[str, Any], float]] = {}
    used_names: set[str] = set()
    for score, cid, sig in triples:
        if cid in assigned or sig["name"] in used_names:
            continue
        assigned[cid] = (sig, score)
        used_names.add(sig["name"])

    out = []
    for cid in centroids.index:
        cid = int(cid)
        sig, score = assigned[cid]
        zrow = zmat.loc[cid]
        row = centroids.loc[cid].fillna(means)
        # alternatives — лучшие прочие шаблоны для этого кластера (справочно)
        alts = sorted(
            (
                (s["name"], _score_archetype(zrow, s["weights"]))
                for s in ARCHETYPE_SIGNATURES
                if s["name"] != sig["name"]
            ),
            key=lambda t: t[1],
            reverse=True,
        )[:2]
        top_pos = zrow.nlargest(3)
        top_neg = zrow.nsmallest(3)
        out.append(
            {
                "cluster_id": cid,
                "name": sig["name"],
                "hint": sig["hint"],
                "score": round(float(score), 2),
                # пометка для прозрачности: уверенно ли шаблон подошёл
                "fit": "strong" if score >= MIN_SIGNATURE_SCORE else "weak",
                "alternatives": [
                    {"name": nm, "score": round(float(s), 2)} for nm, s in alts
                ],
                "top_above_avg": [
                    {"feature": f, "z": round(float(z), 2), "value": round(float(row[f]), 3)}
                    for f, z in top_pos.items()
                ],
                "top_below_avg": [
                    {"feature": f, "z": round(float(z), 2), "value": round(float(row[f]), 3)}
                    for f, z in top_neg.items()
                ],
            }
        )
    return out


def build_matchup_matrix(
    frame: pd.DataFrame, fighter_clusters: dict[str, int], k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает три матрицы k×k:
      - n_fights : сколько боёв cluster_A vs cluster_B (A = f_1)
      - f1_wins : сколько из них выиграл f_1
      - f1_win_rate : f1_wins / n_fights
    Сохраняем как k×k DataFrame с целочисленным индексом 0..k-1.
    """
    # синтетические снапшот-бои (dummy-оппонент) — не реальные бои, исключаем
    real = frame
    if "is_synthetic" in frame.columns:
        real = frame[~frame["is_synthetic"].fillna(False)]
    rows = []
    for _, row in real[["f_1_id", "f_2_id", "target_f1_win"]].dropna().iterrows():
        ca = fighter_clusters.get(row["f_1_id"])
        cb = fighter_clusters.get(row["f_2_id"])
        if ca is None or cb is None:
            continue
        rows.append((ca, cb, int(row["target_f1_win"])))

    df = pd.DataFrame(rows, columns=["c_a", "c_b", "win"])
    n_fights = (
        df.groupby(["c_a", "c_b"]).size().unstack(fill_value=0).reindex(
            index=range(k), columns=range(k), fill_value=0
        )
    )
    f1_wins = (
        df.groupby(["c_a", "c_b"])["win"].sum().unstack(fill_value=0).reindex(
            index=range(k), columns=range(k), fill_value=0
        )
    )
    win_rate = (f1_wins / n_fights.replace(0, np.nan)).fillna(0.0)
    return n_fights, f1_wins, win_rate


def run(k: int = 8) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    frame, features, diff_features, history, profile = load_frame()
    idx = build_fighter_index(frame, history, profile)

    candidates = idx[idx["num_career_fights"] >= MIN_FIGHTS_FOR_CLUSTERING].copy()
    signature = candidates[SIGNATURE_FEATURES].astype(float)

    pipe, labels = fit_clusters(signature, k=k)
    candidates["cluster"] = labels
    fighter_clusters: dict[str, int] = dict(
        zip(candidates["fighter_id"], candidates["cluster"].astype(int))
    )

    # для бойцов с < MIN_FIGHTS — присвоим ближайший кластер по их signature
    rest = idx[~idx["fighter_id"].isin(candidates["fighter_id"])].copy()
    if len(rest):
        rest_sig = rest[SIGNATURE_FEATURES].astype(float)
        rest_labels = pipe.predict(rest_sig.values)
        for fid, lab in zip(rest["fighter_id"], rest_labels):
            fighter_clusters[fid] = int(lab)

    centroids = raw_centroids(signature, labels, k)
    archetypes = name_clusters(centroids)

    n_fights, f1_wins, win_rate = build_matchup_matrix(frame, fighter_clusters, k)

    artifact = {
        "pipeline": pipe,
        "signature_features": SIGNATURE_FEATURES,
        "fighter_clusters": fighter_clusters,
        "archetypes": archetypes,
        "centroids": centroids.to_dict(orient="records"),
        "matchup_n_fights": n_fights.values.tolist(),
        "matchup_f1_wins": f1_wins.values.tolist(),
        "matchup_f1_win_rate": win_rate.values.tolist(),
        "min_fights_for_clustering": MIN_FIGHTS_FOR_CLUSTERING,
        "k": k,
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    summary = {
        "k": k,
        "archetypes": archetypes,
        "cluster_size": (
            candidates.groupby("cluster").size().reindex(range(k), fill_value=0).to_dict()
        ),
        "n_fighters_clustered": len(fighter_clusters),
        "n_fighters_full_signature": int(len(candidates)),
        "matchup_n_fights": n_fights.values.tolist(),
        "matchup_f1_win_rate": win_rate.round(3).values.tolist(),
        "centroids_sample_features": SIGNATURE_FEATURES[:5],
        "centroids_first_3_rows": centroids.head(3).round(3).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_artifact() -> dict[str, Any]:
    return joblib.load(ARTIFACT_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Style clustering + matchup matrix.")
    p.add_argument("--k", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(k=args.k)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
