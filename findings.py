"""Шаг 3. Расчёт всех результатов для приложения и защиты → out/findings.json

1. Карта доказательности: аудит всех пар показателей каталога.
2. Центральное открытие: региональный «парадокс достатка» + проверки устойчивости.
3. Демонстрация ловушки агрегирования (наивная vs контролируемая связь).
4. Разрыв между самооценкой достатка и лишениями (только маргинальные доли).
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import engine as E
from build_dataset import ROOT, income, deprivation, read

OUT = E.OUT


def paired_regional(R, a, b):
    rows = {}
    for y, g in R.groupby("year"):
        rows[int(y)] = float(g[[a, b]].corr(method="spearman").iloc[0, 1])
    return rows


def replication_2022():
    """Независимая репликация на 2022 г. (пункты D002 закодированы так же, как в 2023-2024)."""
    o = read(ROOT / "d002/2022/ocenka.csv").set_index("NOMER")
    s = read(ROOT / "d002/2022/subject.csv").set_index("NOMER")
    df = pd.DataFrame({"region": o.TE, "ritual": (o.GR38 == 2).astype(int),
                       "shock": (o.GR35 == 2).astype(int), "holiday": (o.GR36 == 2).astype(int)})
    df["life_sat"] = s.GR1.where(s.GR1 <= 10)
    inc = income(2022)
    r08 = read(ROOT / "d008/2022/kontr_k.csv").groupby("NOMER").size().rename("hh_size")
    df = df.join(inc[["income_annual"]]).join(r08)
    df["income_pc"] = df.income_annual / df.hh_size / 12
    R = df.groupby("region").mean(numeric_only=True)
    out = {k: float(R[["income_pc", k]].corr(method="spearman").iloc[0, 1]) for k in ["ritual", "shock", "holiday", "life_sat"]}
    out["n_households"] = int(len(df)); out["k_regions"] = int(len(R))
    return out


def main():
    d = E.load()
    rng = np.random.default_rng(E.SEED)
    vars_ = list(E.CATALOG)

    # ---------- 1. карта доказательности
    pairs = []
    for a, b in combinations(vars_, 2):
        r = E.audit_pair(d, a, b, rng)
        pairs.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        pairs[-1]["raw_ci"] = [round(x, 4) for x in r["raw_ci"]]
        pairs[-1]["partial_ci"] = [round(x, 4) for x in r["partial_ci"]]
    # региональные связи для всех пар (быстрая точечная оценка + проверка двух лет)
    R = E.regional_table(d, vars_ + ["dep_severe"])
    for p in pairs:
        reg = paired_regional(R, p["a"], p["b"])
        p["reg"] = {str(k): round(v, 3) for k, v in reg.items()}
    # полные бутстрап-интервалы — для пар, где связь не подтверждена на уровне домохозяйств
    for p in pairs:
        if not p["household_valid"]:
            rp = E.regional_pair(d, p["a"], p["b"], rng)
            p["reg_ci"] = {str(y): [round(c, 3) for c in v["ci"]] for y, v in rp["by_year"].items()}
            p["reg_robust"] = rp["robust"]
        else:
            p["reg_robust"] = None
    levels = pd.Series(["HOUSEHOLD" if p["household_valid"] else ("REGIONAL" if p["reg_robust"] else "NONE") for p in pairs])
    for p, lv in zip(pairs, levels):
        p["level"] = lv

    # ---------- 2. центральное открытие: регионы
    R["name"] = R.region.map(E.REGIONS)
    urban_share = d.groupby(["year", "region"]).urban.mean().rename("urban_share").reset_index()
    R = R.merge(urban_share, on=["year", "region"])
    core = {}
    targets = {"dep_ritual": "Обряды только в долг", "dep_shock": "Нет подушки на 30–40 тыс.",
               "dep_holiday": "Нет отпуска", "dep_severe": "Глубокая депривация (3+ лишений)", "life_sat": "Удовлетворённость жизнью"}
    cities = {71, 75, 79}
    for t in targets:
        rp = E.regional_pair(d, "income_pc", t, rng) if t != "dep_severe" else None
        row = {"label": targets[t], "by_year": {}}
        for y, g in R.groupby("year"):
            rho = float(g[["income_pc", t]].corr(method="spearman").iloc[0, 1])
            loo = [float(g.drop(i)[["income_pc", t]].corr(method="spearman").iloc[0, 1]) for i in g.index]
            nc = g[~g.region.isin(cities)]
            rho_nc = float(nc[["income_pc", t]].corr(method="spearman").iloc[0, 1])
            # частная ранговая корреляция при контроле доли городского населения
            rk = g[["income_pc", t, "urban_share"]].rank()
            res = {}
            for c in ["income_pc", t]:
                X = np.c_[np.ones(len(rk)), rk.urban_share]
                beta = np.linalg.lstsq(X, rk[c], rcond=None)[0]
                res[c] = rk[c] - X @ beta
            rho_part = float(np.corrcoef(res["income_pc"], res[t])[0, 1])
            row["by_year"][str(int(y))] = {"rho": round(rho, 3), "loo_min": round(min(loo), 3), "loo_max": round(max(loo), 3),
                                           "no_cities": round(rho_nc, 3), "partial_urban": round(rho_part, 3)}
            if rp:
                row["by_year"][str(int(y))]["ci"] = [round(c, 3) for c in rp["by_year"][int(y)]["ci"]]
        core[t] = row
    rep = replication_2022()

    regions = []
    for _, r in R.iterrows():
        regions.append({"year": int(r.year), "region": int(r.region), "name": r["name"], "n": int(r.n),
                        "income_pc": round(float(r.income_pc)), "dep_ritual": round(float(r.dep_ritual), 4),
                        "dep_shock": round(float(r.dep_shock), 4), "dep_holiday": round(float(r.dep_holiday), 4),
                        "dep_severe": round(float(r.dep_severe), 4), "life_sat": round(float(r.life_sat), 3),
                        "urban_share": round(float(r.urban_share), 3)})

    # ---------- 3. ловушка агрегирования
    trap = [p for p in pairs if not p["household_valid"] and p["reg_robust"]]
    trap = sorted(trap, key=lambda p: -abs(max(p["reg"].values(), key=abs)))[:8]

    # ---------- 4. разрыв самооценки
    gap = {"self_mid_or_above": round(float((d.self_class >= 3).mean()), 4),
           "self_below_avg": round(float((d.self_class <= 2).mean()), 4),
           "dep_shock": round(float(d.dep_shock.mean()), 4), "dep_ritual": round(float(d.dep_ritual.mean()), 4),
           "dep_holiday": round(float(d.dep_holiday.mean()), 4), "dep_severe": round(float(d.dep_severe.mean()), 4),
           "fies_any": round(float((d.fies_raw >= 1).mean()), 4)}
    dep_items = {k: {"label": v[1], "rate": round(float(d[f"dep_{k}"].mean()), 4)} for k, v in __import__("build_dataset").DEPRIVATION.items()}

    findings = {"meta": {"n_households": int(len(d)), "years": sorted(d.year.unique().tolist()), "k_regions": int(d.region.nunique()),
                         "n_boot": E.N_BOOT, "n_perm": E.N_PERM, "seed": E.SEED},
                "catalog": {k: {"label": v[0], "form": v[1], "kw": v[2]} for k, v in E.CATALOG.items()},
                "pairs": pairs, "level_counts": levels.value_counts().to_dict(),
                "core": core, "replication_2022": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()},
                "regions": regions, "trap": trap, "gap": gap, "dep_items": dep_items}
    (OUT / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=1))
    print("levels:", findings["level_counts"])
    print(json.dumps(core, ensure_ascii=False, indent=1)[:2500])
    print("2022:", findings["replication_2022"]); print("gap:", gap)


if __name__ == "__main__":
    main()
