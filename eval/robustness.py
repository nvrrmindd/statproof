"""Проверки устойчивости вердиктов на уровне семей.

1. Поправка на множественные сравнения (253 пары): аналитическое p для ρ внутри ячеек
   (t-приближение, n − число ячеек степеней свободы) и процедура Бенджамини — Хохберга, q = 0,01.
   Перестановочное p при 200 перестановках не опускается ниже 0,005, поэтому для поправки используется аналитическое.
2. Ключевые пары пересчитаны на 10 000 перестановках внутри ячеек.

    python3 qol_ai/eval/robustness.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine as E  # noqa: E402

HERE = Path(__file__).resolve().parent


def analytic_p(r: float, n: int, k: int) -> float:
    df = max(n - k - 2, 1)
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), df))


def perm_p(d, a, b, B, seed):
    rng = np.random.default_rng(seed)
    m = d[[a, b]].notna().all(axis=1).to_numpy()
    codes = __import__("pandas").factorize(d.loc[m, "cell"])[0]
    ra = E._rank(d.loc[m, a].to_numpy(float)); rb = E._rank(d.loc[m, b].to_numpy(float))
    pa, pb = E._demean(ra, codes), E._demean(rb, codes)
    obs = abs(E._corr(pa, pb))
    order = np.argsort(codes, kind="stable"); groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    ge = 0
    for _ in range(B):
        pb2 = pb.copy()
        for g in groups:
            pb2[g] = pb[rng.permutation(g)]
        ge += abs(E._corr(pa, pb2)) >= obs
    return (ge + 1) / (B + 1), E._corr(pa, pb)


def main():
    f = json.loads((E.OUT / "findings.json").read_text())
    d = E.load()
    k = d["cell"].nunique()
    P = f["pairs"]
    p_an = np.array([analytic_p(p["partial"], p["n"], k) for p in P])
    # Бенджамини — Хохберг, q = 0,01
    q = 0.01
    order = np.argsort(p_an); m = len(P)
    passed = np.zeros(m, bool)
    thr = q * np.arange(1, m + 1) / m
    ok = p_an[order] <= thr
    if ok.any():
        passed[order[: np.max(np.flatnonzero(ok)) + 1]] = True
    hh = [i for i, p in enumerate(P) if p["level"] == "HOUSEHOLD"]
    survive = sum(passed[i] for i in hh)
    worst = max(p_an[i] for i in hh)
    print(f"HOUSEHOLD-связей: {len(hh)}; переживают поправку Бенджамини — Хохберга (q = 0,01): {survive}")
    print(f"наибольшее аналитическое p среди них: {worst:.2e}; порог BH для последней: {q * len(hh) / m:.2e}")
    # 10 000 перестановок для ключевых пар
    key = [("fies_raw", "life_sat"), ("dep_n", "life_sat"), ("income_pc", "dep_ritual"), ("n_children", "income_pc")]
    rows = []
    for a, b in key:
        pp, part = perm_p(d, a, b, 10000, E.SEED)
        rec = next(p for p in P if {p["a"], p["b"]} == {a, b})
        rows.append({"pair": [a, b], "partial": round(part, 4), "p_200": rec["p_perm"], "p_10000": round(pp, 5),
                     "level": rec["level"]})
        print(f"{a} × {b}: ρ внутри {part:+.4f}; p (200) = {rec['p_perm']}; p (10 000) = {pp:.5f}; вердикт {rec['level']}")
    out = {"household_pairs": len(hh), "survive_bh_q001": int(survive), "max_p_analytic_household": worst,
           "permutations_10000": rows}
    (HERE / "robustness_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
