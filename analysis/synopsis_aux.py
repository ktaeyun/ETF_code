# -*- coding: utf-8 -*-
"""
시놉시스 보조 자료 3건.

  1. 실측 최종값의 시뮬 분포 백분위          (콘솔)
  2. PIT 히스토그램 2장                      results/figures/pit_hist_{us,kr}.png / .svg
  3. Base 시나리오 Volatility 절대값         (콘솔)

스타일·색상·폰트는 analysis/fanchart.py 와 동일하다 (나란히 쓰는 자료).
사양: NAV = btc_aligned, T = 342, N = 1000, seed = 42

  US ETF : NAV x (1+GAP)             Step 6-A 검증 대상 (실측 etf_true 대조)
  KR ETF : NAV x (1+GAP) x (1+KP)    Step 6-B / Step 7 (실측 성분곱 대리 계열 대조)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from analysis.fanchart import (BAND, FS_LABEL, FS_TICK, FS_TITLE, LINE_MEDIAN,  # noqa: E402
                               OUT_DIR, load_paths)

N_BINS = 10


# ══════════════════════════════════════════════════════════════
# 1. 실측 최종값의 시뮬 분포 백분위
# ══════════════════════════════════════════════════════════════

def terminal_percentile(d):
    print("=" * 74)
    print("1. 실측 최종값의 시뮬 분포 백분위 (최종 시점, N=1000)")
    print("=" * 74)

    for tag, unit, sim, actual, fmt in [
        ("US ETF   ", "USD", d["us_sim"], d["us_actual"], "{:>12,.2f}"),
        ("Korean ETF", "KRW", d["kr_sim"], d["kr_actual"], "{:>12,.0f}"),
    ]:
        term, obs = sim[:, -1], float(actual[-1])
        pct = float(np.mean(term <= obs) * 100.0)
        qs = {q: float(np.percentile(term, q)) for q in (5, 25, 50, 75, 95)}

        print(f"\n[{tag}] 단위 {unit}")
        print(f"  실측 최종값        {fmt.format(obs)}")
        print(f"  시뮬 분포 백분위   {pct:>11.1f} 백분위")
        print("  시뮬 분포")
        for q, v in qs.items():
            mark = "  <- 실측이 이 근처" if abs(pct - q) < 5 else ""
            print(f"    p{q:<3d}            {fmt.format(v)}{mark}")
    print()


# ══════════════════════════════════════════════════════════════
# 2. PIT 히스토그램
# ══════════════════════════════════════════════════════════════

def pit_values(actual, sim):
    """u_t = ECDF_t(actual_t). metrics.pit_ks_test 와 같은 정의(로그수익률 기준)."""
    a = np.diff(np.log(actual))
    s = np.diff(np.log(sim), axis=1)
    T = min(len(a), s.shape[1])
    return np.array([np.mean(s[:, t] <= a[t]) for t in range(T)])


def pit_hist(u, title, out_stem):
    n = len(u)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(u, bins=N_BINS, range=(0.0, 1.0), color=BAND, alpha=0.75,
            edgecolor="white", linewidth=0.6)
    ax.axhline(n / N_BINS, color=LINE_MEDIAN, lw=1.2, ls="--", dashes=(4, 3))

    ax.set_title(title, fontsize=FS_TITLE, pad=8)
    ax.set_xlabel("PIT value", fontsize=FS_LABEL)
    ax.set_ylabel("Frequency", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))

    ax.grid(axis="y", color="#C9CDD2", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9AA0A6")
        ax.spines[side].set_linewidth(0.6)

    fig.subplots_adjust(left=0.115, right=0.965, top=0.90, bottom=0.145)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"  저장: {path}")
    plt.close(fig)


def pit_section(d):
    print("=" * 74)
    print("2. PIT 히스토그램 (KS 통계량은 콘솔에만 출력)")
    print("=" * 74)

    for tag, sim, actual, title, stem in [
        ("US ETF", d["us_sim"], d["us_actual"], "PIT Histogram — US ETF", "pit_hist_us"),
        ("Korean ETF", d["kr_sim"], d["kr_actual"], "PIT Histogram — Korean ETF", "pit_hist_kr"),
    ]:
        u = pit_values(actual, sim)
        ks_stat, ks_p = kstest(u, "uniform")
        print(f"\n[{tag}]  n={len(u)}  KS={ks_stat:.4f}  p={ks_p:.4f}"
              f"  평균={u.mean():.4f} (기준 0.5)")
        counts, _ = np.histogram(u, bins=N_BINS, range=(0.0, 1.0))
        print("  bin 도수: " + " ".join(f"{c:>3d}" for c in counts)
              + f"   (균등 기준 {len(u) / N_BINS:.1f})")
        pit_hist(u, title, stem)
    print()


# ══════════════════════════════════════════════════════════════
# 3. Base Volatility 절대값
# ══════════════════════════════════════════════════════════════

def base_volatility(d, mc):
    """경로별 로그수익률 표준편차의 중앙값·평균. Step 11 효과크기의 기준값이다."""
    print("=" * 74)
    print("3. Base 시나리오 Volatility 절대값 (경로별 로그수익률 표준편차)")
    print("=" * 74)

    bases = {
        "전체 ETF  (NAV x (1+GAP) x (1+KP))": d["kr_sim"],
        "한국 요인 (      (1+GAP) x (1+KP))": (1.0 + mc["mc_gap"]) * (1.0 + mc["mc_kp"]),
    }
    ann = np.sqrt(252.0)
    print(f"\n  {'기준':<36}{'중앙값(일별)':>14}{'평균(일별)':>13}"
          f"{'중앙값(연율)':>14}{'평균(연율)':>13}")
    out = {}
    for label, paths in bases.items():
        v = np.diff(np.log(np.maximum(paths, 1e-12)), axis=1).std(axis=1)
        med, mean = float(np.median(v)), float(np.mean(v))
        out[label] = (med, mean)
        print(f"  {label:<36}{med:>14.6f}{mean:>13.6f}"
              f"{med * ann:>14.4f}{mean * ann:>13.4f}")

    (m_etf, a_etf), (m_kr, a_kr) = out.values()
    print(f"\n  비율 전체/한국   중앙값 {m_etf / m_kr:.2f}배   평균 {a_etf / a_kr:.2f}배")
    print()


def main():
    print("[경로 로드]")
    d, mc = load_paths()
    print()
    terminal_percentile(d)
    pit_section(d)
    base_volatility(d, mc)


if __name__ == "__main__":
    main()
