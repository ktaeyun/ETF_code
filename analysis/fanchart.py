# -*- coding: utf-8 -*-
"""
시놉시스 PPT용 팬차트 2종 생성.

  results/figures/fanchart_us.png / .svg   미국 ETF (USD)  : NAV x (1+GAP)
  results/figures/fanchart_kr.png / .svg   한국형 ETF (KRW): NAV x (1+GAP) x (1+KP)

경로 배열은 Phase 1 캐시(results/simulator/cache/sim_arrays.npz)에서 읽는다.
캐시가 없거나 사양이 다르면 simulator/main.py 를 먼저 실행할 것.

사양: NAV = btc_aligned, T = 342, N = 1000, seed = 42
좌우 나란히 배치할 한 쌍이므로 두 그림의 스타일을 완전히 동일하게 맞춘다.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from simulator.arima_garch_t_nav_simulator import log_returns_to_nav  # noqa: E402
from simulator.data_loader import (align_to_nav_grid, load_gap_exog,  # noqa: E402
                                   load_kp_exog, load_nav_exog_and_returns)

CACHE_DIR = _ROOT / "results" / "simulator" / "cache"
OUT_DIR = _ROOT / "results" / "figures"
ANCHOR_KRW = 10_000.0

# ── 스타일 (두 그림 공통) ────────────────────────────────
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 11, 9, 8, 8, 8
BAND = "#4C78A8"          # 밴드 단색 계열
LINE_MEDIAN = "#1F3B57"   # 중앙값 (밴드와 같은 계열, 진하게)
LINE_ACTUAL = "#E4572E"   # 실측 (대비색)
LW_MEDIAN, LW_ACTUAL = 1.4, 1.4
A_OUTER, A_INNER = 0.20, 0.38


def _thousands(x, _pos):
    """천 단위 구분 기호. 축약하지 않는다."""
    return f"{x:,.0f}"


def load_paths():
    """캐시에서 MC 경로와 실측 계열을 읽어 두 그림의 입력을 만든다."""
    npz, meta = CACHE_DIR / "sim_arrays.npz", CACHE_DIR / "sim_meta.json"
    if not npz.exists() or not meta.exists():
        raise FileNotFoundError(
            f"Phase 1 캐시 없음: {npz}\n  python simulator/main.py 를 먼저 실행하세요.")

    m = json.loads(meta.read_text(encoding="utf-8"))
    nav_source, S0 = m.get("nav_source", "btc"), float(m["S0"])
    arr = np.load(str(npz))
    mc_nav, mc_gap, mc_kp = arr["mc_nav"], arr["mc_gap"], arr["mc_kp"]

    nav_df = load_nav_exog_and_returns(base_dir=str(_ROOT), nav_source=nav_source)
    gap_df, kp_df = align_to_nav_grid(
        nav_df, load_gap_exog(base_dir=str(_ROOT)), load_kp_exog(base_dir=str(_ROOT)))

    T = min(mc_nav.shape[1], mc_gap.shape[1], mc_kp.shape[1], len(nav_df))
    dates = pd.to_datetime(nav_df["Date"].to_numpy()[:T])

    # 미국 ETF: NAV x (1+GAP), 실측 대응물은 etf_true
    us_sim = mc_nav[:, :T] * (1.0 + mc_gap[:, :T])
    us_actual = nav_df["etf_true"].to_numpy(dtype=float)[:T]

    # 한국형 ETF: 세 성분 곱을 경로별로 10,000원에 앵커링
    actual_nav = np.asarray(
        log_returns_to_nav(pd.Series(nav_df["Log Return"].to_numpy(dtype=float)), S0=S0)
    ).flatten()[:T]
    kr_actual_raw = (actual_nav
                     * (1.0 + gap_df["etf_premium"].to_numpy(dtype=float)[:T])
                     * (1.0 + kp_df["Kimchi Premium"].to_numpy(dtype=float)[:T]))
    kr_sim_raw = us_sim * (1.0 + mc_kp[:, :T])

    def anchor(a):
        first = a[..., :1]
        return np.where(first != 0, a * (ANCHOR_KRW / np.where(first == 0, 1.0, first)), a)

    print(f"  사양 {nav_source}  N={mc_nav.shape[0]}  T={T}  S0={S0}"
          f"  {dates[0].date()} ~ {dates[-1].date()}")
    return (dict(dates=dates,
                 us_sim=us_sim, us_actual=us_actual,
                 kr_sim=anchor(kr_sim_raw), kr_actual=anchor(kr_actual_raw)),
            {"mc_gap": mc_gap[:, :T], "mc_kp": mc_kp[:, :T]})


def fanchart(dates, sim, actual, title, ylabel, actual_label, out_stem, decimals=0):
    """시점별 분위수 밴드 + 중앙값 (+ 실측선).

    actual=None 이면 실측선·범례·라벨을 모두 생략한다. 한국형 ETF는 실재하지 않아
    대조할 관측 계열이 없으므로 밴드와 중앙값만 그린다.
    """
    q05, q25, q50, q75, q95 = (np.percentile(sim, q, axis=0) for q in (5, 25, 50, 75, 95))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(dates, q05, q95, color=BAND, alpha=A_OUTER, linewidth=0)
    ax.fill_between(dates, q25, q75, color=BAND, alpha=A_INNER, linewidth=0)
    ax.plot(dates, q50, color=LINE_MEDIAN, lw=LW_MEDIAN, solid_capstyle="round")
    if actual is not None:
        ax.plot(dates, actual, color=LINE_ACTUAL, lw=LW_ACTUAL, solid_capstyle="round")

    ax.set_title(title, fontsize=FS_TITLE, pad=8)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.grid(axis="y", color="#C9CDD2", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9AA0A6")
        ax.spines[side].set_linewidth(0.6)

    # 최종 시점 값 라벨. 두 개일 때 겹치면 위아래로 밀어 놓는다.
    x_end = dates[-1]
    labels = [(float(q50[-1]), LINE_MEDIAN)]
    if actual is not None:
        labels.append((float(actual[-1]), LINE_ACTUAL))
    labels.sort()
    span = float(np.nanmax(q95) - np.nanmin(q05))
    offsets = [v for v, _ in labels]
    if len(labels) == 2 and labels[1][0] - labels[0][0] < 0.06 * span:
        mid = 0.5 * (labels[0][0] + labels[1][0])
        offsets = [mid - 0.035 * span, mid + 0.035 * span]
    for (val, color), y in zip(labels, offsets):
        ax.annotate(format(val, f",.{decimals}f"),
                    xy=(x_end, val), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=FS_ANNOT, color=color, annotation_clip=False)

    handles = [
        Patch(facecolor=BAND, alpha=A_OUTER, label="5–95%"),
        Patch(facecolor=BAND, alpha=A_INNER, label="25–75%"),
        Line2D([], [], color=LINE_MEDIAN, lw=LW_MEDIAN, label="Median"),
    ]
    if actual is not None:
        handles.append(Line2D([], [], color=LINE_ACTUAL, lw=LW_ACTUAL, label=actual_label))
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left",
              frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.35)

    fig.subplots_adjust(left=0.155, right=0.865, top=0.90, bottom=0.135)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, facecolor="white")
        print(f"  저장: {path}")
    plt.close(fig)


def base_volatility(d, mc):
    """Base 시나리오의 Volatility 절대값 (경로별 로그수익률 표준편차의 평균).

    Step 11의 효과크기가 Base 대비 상대값이라 기준값이 필요하다.
    정의는 compute_risk_metrics(scenario_main.py)와 같고, 앵커링은 로그수익률
    표준편차에 영향을 주지 않으므로 원경로에서 바로 계산한다.

      etf : NAV x (1+GAP) x (1+KP)   투자자가 보는 가격
      kr  :       (1+GAP) x (1+KP)   조작 대상만 분리한 한국 시장 요인
    """
    def vol(a):
        r = np.diff(np.log(np.maximum(a, 1e-12)), axis=1)
        return float(np.nanmean(r.std(axis=1)))

    etf_paths = d["us_sim"] * (1.0 + mc["mc_kp"])
    kr_paths = (1.0 + mc["mc_gap"]) * (1.0 + mc["mc_kp"])

    v_etf, v_kr = vol(etf_paths), vol(kr_paths)
    print("\n[3] Base Volatility 절대값 (일별 로그수익률 표준편차, 무단위)")
    print("  기준          일별        연율화(x sqrt(252))")
    print(f"  etf (전체)   {v_etf:.6f}    {v_etf * np.sqrt(252):.4f}")
    print(f"  kr  (한국)   {v_kr:.6f}    {v_kr * np.sqrt(252):.4f}")
    print(f"  비율 etf/kr  {v_etf / v_kr:.2f}배")
    return v_etf, v_kr


def main():
    print("[1] 경로 로드")
    d, mc = load_paths()

    print("[2] 팬차트 생성")
    fanchart(d["dates"], d["us_sim"], d["us_actual"],
             "Simulated vs. Actual US Spot Bitcoin ETF", "Price (USD)",
             "Actual", "fanchart_us", decimals=1)
    # 한국형 ETF는 실재하지 않으므로 대조선을 그리지 않는다 (밴드 + 중앙값만).
    fanchart(d["dates"], d["kr_sim"], None,
             "Simulated Korean Spot Bitcoin ETF", "Price (KRW)",
             None, "fanchart_kr")

    base_volatility(d, mc)


if __name__ == "__main__":
    main()
