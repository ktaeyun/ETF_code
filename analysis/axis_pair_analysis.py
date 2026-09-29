# -*- coding: utf-8 -*-
"""
한 축만 다른 시나리오 쌍으로 축별 효과 방향을 식별한다 (이론 가설 검증용).

    python analysis/axis_pair_analysis.py <작업공간> <출력폴더> [--basis kr] [--metric Volatility]

시뮬레이션된 시나리오 전체(effect_size_matrix 열)에서 5축 중 정확히 한 축만 다른 쌍을 모두
찾고, 쌍마다 "높은 수준 − 낮은 수준" 효과 차이를 계산한다. 축별로 쌍 수, 방향 일치율,
차이의 범위·중앙값, 부호검정 p값을 낸다. Base 대비 부호에 의존하지 않는 시나리오 간 비교라
표본 구성 효과에 영향을 받지 않는다.

출력
    tables/axis_pairs_<basis>_<metric>.csv     쌍 목록
    tables/axis_summary_<basis>_<metric>.csv   축별 요약
    figures/fig10_axis_pairs.png               축별 쌍 차이 분포
"""
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

AX = ["Bitcoin_RV", "VKOSPI", "KR_Volume", "KR_SVI", "Global_SVI"]
ORDER = {"Low": 0, "Mid": 1, "High": 2, "Normal": 0, "Extreme": 1}
# 이론 예측 부호 (H1~H5). None = 이론이 갈림
HYPOTHESIS = {"KR_SVI": +1, "VKOSPI": +1, "Bitcoin_RV": +1, "Global_SVI": +1, "KR_Volume": None}
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
AX_COLOR = dict(zip(AX, [VIOLET, ORANGE, BLUE, AQUA, YELLOW]))


def load(ws: Path, basis: str, metric: str):
    scen = pd.read_csv(ws / "results" / "scenario_selection" / "final_scenarios_latest.csv").set_index("Scenario_ID")
    eff = pd.read_csv(ws / "results" / "scenario_simulator" / "significance_test" /
                      f"effect_size_matrix_pct_{basis}.csv", index_col=0)
    ids = [c for c in eff.columns if c in scen.index]
    scen = scen.loc[ids]
    scen["effect"] = eff.loc[metric, ids].values
    return scen


def pairs(scen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(scen.index, 2):
        diff = [x for x in AX if scen.loc[a, x] != scen.loc[b, x]]
        if len(diff) != 1:
            continue
        x = diff[0]
        lo, hi = (a, b) if ORDER[scen.loc[a, x]] < ORDER[scen.loc[b, x]] else (b, a)
        rows.append({
            "axis": x, "low_id": lo, "high_id": hi,
            "low_level": scen.loc[lo, x], "high_level": scen.loc[hi, x],
            "context": "-".join(str(scen.loc[lo, y]) if y != x else "*" for y in AX),
            "low_freq": int(scen.loc[lo, "joint_frequency"]), "high_freq": int(scen.loc[hi, "joint_frequency"]),
            "low_effect": scen.loc[lo, "effect"], "high_effect": scen.loc[hi, "effect"],
            "diff_pp": scen.loc[hi, "effect"] - scen.loc[lo, "effect"],
        })
    return pd.DataFrame(rows)


def summary(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x in AX:
        d = p.loc[p["axis"] == x, "diff_pp"]
        n = len(d)
        pos = int((d > 0).sum())
        row = {"axis": x, "n_pairs": n, "n_positive": pos, "n_negative": int((d < 0).sum()),
               "median_diff_pp": d.median() if n else np.nan,
               "min_diff_pp": d.min() if n else np.nan, "max_diff_pp": d.max() if n else np.nan,
               "sign_test_p": binomtest(pos, n, 0.5).pvalue if n else np.nan,
               "wilcoxon_p": wilcoxon(d).pvalue if n >= 5 and (d != 0).any() else np.nan}
        h = HYPOTHESIS[x]
        if n == 0:
            row["verdict"] = "not identifiable (no pairs)"
        elif h is None:
            row["verdict"] = f"theory ambiguous; data → {'+' if pos > n / 2 else '-'} ({pos}/{n})"
        else:
            agree = pos if h > 0 else n - pos
            row["verdict"] = f"H predicts {'+' if h > 0 else '−'}; agree {agree}/{n}"
        rows.append(row)
    return pd.DataFrame(rows)


def figure(p: pd.DataFrame, s: pd.DataFrame, out: Path, metric: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    rng = np.random.default_rng(0)
    for i, x in enumerate(AX):
        d = p.loc[p["axis"] == x, "diff_pp"].to_numpy()
        if len(d) == 0:
            ax.text(i, 0, "no pairs\n(not identifiable)", ha="center", va="center", fontsize=8.5, color=MUTED)
            continue
        jitter = (rng.random(len(d)) - 0.5) * 0.35
        ax.scatter(np.full(len(d), i) + jitter, d, s=34, color=AX_COLOR[x], alpha=0.85, edgecolor="white", lw=0.8, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color=INK, lw=1.6, zorder=4)
        r = s.loc[s["axis"] == x].iloc[0]
        # 우세한 방향의 개수로 표기한다 (축마다 이론 방향이 달라 "양수 개수" 고정은 오해를 준다)
        n_dom, sign = (int(r["n_positive"]), "+") if r["n_positive"] >= r["n_negative"]             else (int(r["n_negative"]), "−")
        ax.text(i, max(d.max(), 0) + 0.6, f"{n_dom}/{int(r['n_pairs'])} {sign}",
                ha="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(AX)), AX, fontsize=9)
    ax.set(ylabel=f"{metric} effect, higher − lower level (%p)",
           title="Axis direction from one-axis-different scenario pairs")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.margins(y=0.2)
    ax.text(0.01, 0.01, "Each dot = one pair with the other four axes identical; bar = median; "
            "label = pairs sharing the dominant direction",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / f"fig10_axis_pairs_{metric}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("out")
    ap.add_argument("--basis", default="kr")
    ap.add_argument("--metric", default="Volatility")
    a = ap.parse_args()
    ws, out = Path(a.workspace).resolve(), Path(a.out).resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    if a.metric == "Max_DD_mean":  # 낙폭 효과는 변동성과 거울상: + = 얕아짐
        for k, v in HYPOTHESIS.items():
            HYPOTHESIS[k] = -v if v is not None else None
    scen = load(ws, a.basis, a.metric)
    p = pairs(scen).sort_values(["axis", "diff_pp"])
    s = summary(p)
    p.to_csv(out / "tables" / f"axis_pairs_{a.basis}_{a.metric}.csv", index=False, encoding="utf-8-sig")
    s.to_csv(out / "tables" / f"axis_summary_{a.basis}_{a.metric}.csv", index=False, encoding="utf-8-sig")
    figure(p, s, out / "figures", a.metric)
    print(f"scenarios={len(scen)}  pairs={len(p)}")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
