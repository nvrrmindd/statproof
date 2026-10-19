"""Универсальный движок Provable: любая таблица микроданных.

Строка таблицы — единица наблюдения (семья, человек, предприятие, школа…).
Группы — любые категориальные столбцы (регион, тип поселения, отрасль, год…).

Для каждой пары числовых показателей:
  1. ρ по всем единицам (ранговая корреляция Спирмена);
  2. ρ внутри ячеек — сочетаний всех групповых столбцов (вычитаются средние рангов по ячейкам);
  3. перестановочный тест внутри ячеек (B перестановок, общих для всех пар) и бутстрап-ДИ;
  4. ρ между агрегатами — средними по «группе сравнения» (например, по регионам), бутстрап по группам;
     если задан столбец «повторения» (например, год), связь между агрегатами должна повториться в каждом его значении.

Вердикт:
  UNIT  — связь подтверждена на уровне единиц (p < 0,01, связь сохраняется внутри ячеек);
  GROUP — на уровне единиц нет, но между группами связь устойчива (интервал не включает ноль, знак повторяется);
  NONE  — вывод не делается.
Поправка на множественные сравнения: Бенджамини — Хохберг (q = 0,01) по аналитическому p.
"""
from __future__ import annotations

import math
import zlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261012


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy(float)


def _demean(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    s = np.bincount(codes, weights=x)
    n = np.bincount(codes)
    return x - (s / np.maximum(n, 1))[codes]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def numeric_columns(df: pd.DataFrame, min_unique: int = 3) -> list[str]:
    out = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.5 and s.nunique() >= min_unique:
            out.append(c)
    return out


def audit_table(df: pd.DataFrame, variables: list[str], cell_groups: list[str], compare_group: str,
                repeat_by: str | None = None, n_perm: int = 200, n_boot: int = 200, seed: int = SEED,
                labels: dict | None = None) -> dict:
    """Полный аудит всех пар показателей таблицы. Возвращает отчёт (dict, готов к JSON)."""
    rng = np.random.default_rng(seed)
    labels = labels or {}
    d = df.copy()
    for v in variables:
        d[v] = pd.to_numeric(d[v], errors="coerce")
    d = d.dropna(subset=cell_groups + [compare_group] + ([repeat_by] if repeat_by else []))
    cell = d[cell_groups].astype(str).agg("|".join, axis=1)
    codes_all = pd.factorize(cell)[0]
    n_all = len(d)

    # общие перестановки внутри ячеек: для каждой — индексы, переставляющие строки внутри своей ячейки
    base = np.lexsort((np.arange(n_all), codes_all))
    perms = []
    for _ in range(n_perm):
        shuffled = np.lexsort((rng.random(n_all), codes_all))
        p = np.empty(n_all, dtype=np.int64)
        p[base] = shuffled
        perms.append(p)
    perms = np.array(perms)

    # ранги и центрированные ранги по каждому показателю (на всех строках; пропуски обрабатываются в паре)
    pairs = []
    k_cells = int(codes_all.max()) + 1 if n_all else 0
    reps = sorted(d[repeat_by].astype(str).unique()) if repeat_by else ["все"]
    gvals = d[compare_group].astype(str).to_numpy()
    rvals = d[repeat_by].astype(str).to_numpy() if repeat_by else np.array(["все"] * n_all)

    for a, b in combinations(variables, 2):
        # своя последовательность случайных чисел для каждой пары: вердикт не зависит от порядка столбцов
        key = sorted([str(a), str(b)])
        prng = np.random.default_rng([seed, zlib.crc32(key[0].encode()), zlib.crc32(key[1].encode())])
        m = d[[a, b]].notna().all(axis=1).to_numpy()
        n = int(m.sum())
        if n < 30:
            continue
        codes = pd.factorize(cell[m])[0]
        ra, rb = _rank(d.loc[m, a].to_numpy()), _rank(d.loc[m, b].to_numpy())
        raw = _corr(ra, rb)
        pa, pb = _demean(ra, codes), _demean(rb, codes)
        part = _corr(pa, pb)
        # перестановки: для пар с пропусками строим их на подвыборке (дешёвый вариант — свои перестановки)
        if n == n_all:
            P = perms
        else:
            bsub = np.lexsort((np.arange(n), codes))
            P = []
            for _ in range(n_perm):
                sh = np.lexsort((prng.random(n), codes)); q = np.empty(n, dtype=np.int64); q[bsub] = sh; P.append(q)
            P = np.array(P)
        pbz = pb - pb.mean(); paz = pa - pa.mean()
        den = math.sqrt(float((paz * paz).sum() * (pbz * pbz).sum())) or 1.0
        null = np.abs((pbz[P] @ paz) / den)
        p_perm = float((1 + (null >= abs(part)).sum()) / (n_perm + 1))
        df_t = max(n - k_cells - 2, 1)
        t = part * math.sqrt(df_t / max(1e-12, 1 - part * part))
        p_an = float(2 * stats.t.sf(abs(t), df_t))
        # бутстрап-ДИ для ρ внутри ячеек
        bi = prng.integers(0, n, size=(n_boot, n))
        bs = [_corr(pa[i], pb[i]) for i in bi[: min(n_boot, 200)]]
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        retained = abs(part) / abs(raw) if abs(raw) > 1e-9 else 0.0
        unit_ok = p_perm < 0.01 and abs(part) >= 0.03 and (retained >= 0.5 or abs(part) >= 0.1)

        # уровень групп: средние по группе сравнения, отдельно для каждого значения «повторения»
        sub = pd.DataFrame({"g": gvals[m], "r": rvals[m], "a": d.loc[m, a].to_numpy(), "b": d.loc[m, b].to_numpy()})
        grp = {}
        for r, gdf in sub.groupby("r"):
            means = gdf.groupby("g")[["a", "b"]].mean()
            if len(means) < 5:
                continue
            ga, gb = _rank(means["a"].to_numpy()), _rank(means["b"].to_numpy())
            rho_g = _corr(ga, gb)
            G = len(means)
            idx = prng.integers(0, G, size=(n_boot, G))
            boots = []
            for row in idx:
                x = means["a"].to_numpy()[row] + prng.normal(0, 1e-9, G)
                y = means["b"].to_numpy()[row] + prng.normal(0, 1e-9, G)
                boots.append(_corr(_rank(x), _rank(y)))
            grp[str(r)] = {"rho": rho_g, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "groups": G}
        signs = {np.sign(v["rho"]) for v in grp.values()}
        group_ok = bool(grp) and len(signs) == 1 and all(v["ci"][0] > 0 or v["ci"][1] < 0 for v in grp.values())
        level = "UNIT" if unit_ok else ("GROUP" if group_ok else "NONE")
        pairs.append({"a": a, "b": b, "n": n, "raw": raw, "partial": part, "ci": ci, "p_perm": p_perm, "p_analytic": p_an,
                      "group": grp, "level": level})

    # поправка Бенджамини — Хохберга по аналитическому p
    if pairs:
        p = np.array([x["p_analytic"] for x in pairs]); order = np.argsort(p); mtot = len(p)
        ok = p[order] <= 0.01 * np.arange(1, mtot + 1) / mtot
        passed = np.zeros(mtot, bool)
        if ok.any():
            passed[order[: np.max(np.flatnonzero(ok)) + 1]] = True
        for x, f in zip(pairs, passed):
            x["bh_ok"] = bool(f)
            if x["level"] == "UNIT" and not f:
                x["level"] = "GROUP" if x["group"] and all(v["ci"][0] > 0 or v["ci"][1] < 0 for v in x["group"].values()) else "NONE"
                x["note"] = "не прошла поправку на множественные сравнения"

    counts = {lv: sum(x["level"] == lv for x in pairs) for lv in ("UNIT", "GROUP", "NONE")}
    return {"rows": n_all, "variables": variables, "labels": {v: labels.get(v, v) for v in variables},
            "cell_groups": cell_groups, "compare_group": compare_group, "repeat_by": repeat_by, "repeats": reps,
            "cells": k_cells, "n_perm": n_perm, "n_boot": n_boot, "seed": seed, "pairs": pairs, "counts": counts}
