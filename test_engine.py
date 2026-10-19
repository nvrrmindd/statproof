"""Проверки движка на данных с известным ответом.  python3 qol_ai/test_engine.py"""
import numpy as np
import pandas as pd
import engine as E


def fake(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 20, n)
    reg_eff = rng.normal(0, 1, 20)[region]
    df = pd.DataFrame({"year": rng.integers(2023, 2025, n), "region": region, "urban": rng.integers(0, 2, n)})
    df["cell"] = df.year.astype(str) + "_" + df.region.astype(str) + "_" + df.urban.astype(str)
    hidden = rng.normal(0, 1, n)
    df["x_true"] = hidden + rng.normal(0, 1, n)
    df["y_true"] = hidden + rng.normal(0, 1, n)             # настоящая связь между семьями
    df["x_eco"] = reg_eff + rng.normal(0, 1, n)            # связаны только через регион
    df["y_eco"] = reg_eff + rng.normal(0, 1, n)
    df["noise"] = rng.normal(0, 1, n)
    return df


def test_detects_household_link():
    r = E.audit_pair(fake(), "x_true", "y_true")
    assert r["household_valid"] and r["partial"] > 0.4, r


def test_rejects_ecological_link_at_household_level():
    d = fake()
    r = E.audit_pair(d, "x_eco", "y_eco")
    assert r["raw"] > 0.3 and abs(r["partial"]) < 0.05 and not r["household_valid"], r
    rg = E.regional_pair(d, "x_eco", "y_eco")
    assert all(v["rho"] > 0.7 for v in rg["by_year"].values()) and rg["robust"], rg


def test_rejects_noise():
    d = fake()
    assert not E.audit_pair(d, "noise", "x_true")["household_valid"]
    assert not E.regional_pair(d, "noise", "x_true")["robust"]


def test_grounding_blocks_invented_numbers():
    facts = {"claims": [{"value": 0.848}, {"value": -0.617}], "verdict": "ρ = +0.85 в 2023"}
    assert E.grounding_check("Связь сильная: ρ = 0.85 в 2023 году.", facts) == []
    assert E.grounding_check("Доход вырос на 37%.", facts) == ["37"]


def test_parser():
    assert E.parse_question("Связан ли доход с обрядами?") == ["income_pc", "dep_ritual"]
    assert len(E.parse_question("Как погода влияет на урожай?")) < 2


def test_generic_engine_any_table():
    import generic as G
    d = fake()
    rep = G.audit_table(d, ["x_true", "y_true", "x_eco", "y_eco", "noise"], ["region", "urban", "year"], "region", "year")
    L = {frozenset((p["a"], p["b"])): p["level"] for p in rep["pairs"]}
    assert L[frozenset(("x_true", "y_true"))] == "UNIT", L
    assert L[frozenset(("x_eco", "y_eco"))] == "GROUP", L
    assert L[frozenset(("noise", "x_true"))] == "NONE", L
    rep2 = G.audit_table(d, ["noise", "y_eco", "x_eco", "y_true", "x_true"], ["region", "urban", "year"], "region", "year")
    assert {frozenset((p["a"], p["b"])): p["level"] for p in rep2["pairs"]} == L   # порядок столбцов не влияет


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK ", name)
