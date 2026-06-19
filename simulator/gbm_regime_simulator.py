"""
시나리오별 레짐 조건부 ETF 가격경로 시뮬레이터
=====================================================
입력:
  - results/scenario_selection/final_scenarios_latest.csv
      시나리오 = 5개 외생변수 레짐의 공동 조합
      (Bitcoin_RV, VKOSPI, KR_Volume, KR_SVI, Global_SVI)

파이프라인:
  1. 시나리오 테이블 로드 (scenario_selection.py 결과)
  2. HMM 파이프라인으로 일별 레짐 레이블 재구성
  3. 시나리오별 해당 날짜 필터 → GAP/KP sigma 조건부 추정
  4. 전체 표본으로 OU 기본 파라미터(kappa, mu) 적합
  5. 시나리오별 sigma 주입 → OU GAP/KP 경로 시뮬레이션
  6. ETF = S0 * (1 + GAP_t) * (1 + KP_t) 가격경로 생성
  7. 리스크 지표 계산 + 시각화 저장
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.pairwise_regime_cooccurrence import build_regime_df
from analysis.scenario_selection import (
    _make_label_maps,
    _build_daily_labeled,
    SCENARIO_DIR,
)
from simulator.gap_ou_simulator import fit_ou_basic
from simulator.kp_threshold_ou_simulator import fit_ou_regime

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

SCENARIO_CSV   = SCENARIO_DIR / "final_scenarios_latest.csv"
DISPLAY_COLS   = ["Bitcoin_RV", "VKOSPI", "KR_Volume", "KR_SVI", "Global_SVI"]
OUT_DIR        = _ROOT / "results" / "simulator" / "scenario_regime"

# 시나리오 유형별 색상
_TYPE_COLORS   = {"Core": "steelblue", "Supplemental": "darkorange"}


# ──────────────────────────────────────────────────────────────
# 1. 시나리오 테이블 로드
# ──────────────────────────────────────────────────────────────

def load_scenarios() -> pd.DataFrame:
    """scenario_selection.py 최종 결과 로드."""
    if not SCENARIO_CSV.exists():
        raise FileNotFoundError(
            f"시나리오 파일 없음: {SCENARIO_CSV}\n"
            "python analysis/scenario_selection.py 를 먼저 실행하세요."
        )
    df = pd.read_csv(SCENARIO_CSV)
    # 실제 관측 데이터가 있는 시나리오만 사용
    df = df[df["combo_label"] != "-"].reset_index(drop=True)
    print(f"[시나리오] {len(df)}개 로드: {list(df['Scenario_ID'])}")
    return df


# ──────────────────────────────────────────────────────────────
# 2. 일별 레짐 레이블 DataFrame 구축
# ──────────────────────────────────────────────────────────────

def build_daily_labeled_df(n_init: int = 10, B: int = 500) -> pd.DataFrame:
    """HMM 파이프라인 → 5변수 일별 의미적 레이블 DataFrame.

    Returns
    -------
    pd.DataFrame : DatetimeIndex, 컬럼 = DISPLAY_COLS (문자열 레이블)
    """
    print("[Step 1] HMM 레짐 레이블 재구성...")
    df_regime   = build_regime_df(n_init=n_init, B=B)
    label_maps  = _make_label_maps(df_regime)
    df_labeled  = _build_daily_labeled(df_regime, label_maps)
    print(f"    완전 관측 기간: {df_labeled.index[0].date()} ~ {df_labeled.index[-1].date()}")
    print(f"    총 {len(df_labeled)}일")
    return df_labeled


# ──────────────────────────────────────────────────────────────
# 3. 시나리오별 날짜 필터 & sigma 추정
# ──────────────────────────────────────────────────────────────

def _get_scenario_dates(
    df_labeled: pd.DataFrame,
    scenario_row: pd.Series,
) -> pd.DatetimeIndex:
    """5변수 레짐 레이블이 모두 일치하는 날짜 반환."""
    mask = pd.Series(True, index=df_labeled.index)
    for col in DISPLAY_COLS:
        if col in scenario_row.index and scenario_row[col] != "-":
            mask &= df_labeled[col] == scenario_row[col]
    return df_labeled.index[mask]


def estimate_sigma_per_scenario(
    scenarios_df: pd.DataFrame,
    df_labeled: pd.DataFrame,
    gap_series: pd.Series,
    kp_series: pd.Series,
) -> pd.DataFrame:
    """시나리오별 GAP/KP sigma 추정 (일별 변화량 표준편차).

    GAP sigma: etf_premium 일별 차분의 std
    KP  sigma: Kimchi Premium 일별 차분의 std

    데이터 부족 시 전체 표본 sigma 사용.
    """
    gap_diff_all = gap_series.diff().dropna()
    kp_diff_all  = kp_series.diff().dropna()
    sigma_gap_full = float(gap_diff_all.std())
    sigma_kp_full  = float(kp_diff_all.std())

    rows = []
    for _, sc in scenarios_df.iterrows():
        dates = _get_scenario_dates(df_labeled, sc)

        # GAP sigma
        gap_common = gap_series.index.intersection(dates)
        gap_sub    = gap_series.loc[gap_common].diff().dropna()
        if len(gap_sub) > 5:
            sig_gap  = float(gap_sub.std())
            gap_note = ""
        else:
            sig_gap  = sigma_gap_full
            gap_note = "(fallback)"

        # KP sigma
        kp_common = kp_series.index.intersection(dates)
        kp_sub    = kp_series.loc[kp_common].diff().dropna()
        if len(kp_sub) > 5:
            sig_kp  = float(kp_sub.std())
            kp_note = ""
        else:
            sig_kp  = sigma_kp_full
            kp_note = "(fallback)"

        rows.append({
            "Scenario_ID"  : sc["Scenario_ID"],
            "Type"         : sc["Type"],
            "combo_label"  : sc["combo_label"],
            "n_dates"      : len(dates),
            "n_gap_obs"    : len(gap_sub),
            "n_kp_obs"     : len(kp_sub),
            "sigma_gap_daily" : round(sig_gap, 8),
            "sigma_kp_daily"  : round(sig_kp, 8),
            "sigma_gap_ann"   : round(sig_gap * np.sqrt(252), 6),
            "sigma_kp_ann"    : round(sig_kp * np.sqrt(252), 6),
            "gap_note"     : gap_note,
            "kp_note"      : kp_note,
        })

    df_sigma = pd.DataFrame(rows)
    print("\n=== 시나리오별 Sigma 추정 결과 ===")
    display_cols = [
        "Scenario_ID", "Type", "combo_label", "n_dates",
        "n_gap_obs", "sigma_gap_ann", "gap_note",
        "n_kp_obs",  "sigma_kp_ann",  "kp_note",
    ]
    print(df_sigma[display_cols].to_string(index=False))
    return df_sigma


# ──────────────────────────────────────────────────────────────
# 4. OU 기본 파라미터 전체 표본 적합
# ──────────────────────────────────────────────────────────────

def fit_base_ou_params(
    gap_series: pd.Series,
    kp_series: pd.Series,
) -> Tuple[dict, dict]:
    """전체 표본으로 GAP / KP OU 기본 파라미터 적합.

    시뮬레이션에서 kappa, mu는 전체 표본값을 사용하고
    sigma0만 시나리오별 값으로 교체한다.
    """
    gap_ou = fit_ou_basic(gap_series)
    kp_ou  = fit_ou_basic(kp_series)
    print(f"\n[기본 OU 파라미터 (전체 표본)]")
    print(f"  GAP: kappa={gap_ou['kappa']:.4f}  mu={gap_ou['mu']:.6f}  sigma0={gap_ou['sigma0']:.6f}")
    print(f"  KP : kappa={kp_ou['kappa']:.4f}  mu={kp_ou['mu']:.6f}  sigma0={kp_ou['sigma0']:.6f}")
    return gap_ou, kp_ou


# ──────────────────────────────────────────────────────────────
# 5. OU 경로 시뮬레이션 (sigma 주입)
# ──────────────────────────────────────────────────────────────

def simulate_ou_paths(
    base_params: dict,
    sigma0_override: float,
    x0: float,
    T: int,
    N: int,
    seed: int = 42,
) -> np.ndarray:
    """OU 경로 N개 시뮬레이션, sigma0만 시나리오 값으로 교체.

    X(t+1) = X(t) + kappa*(mu - X(t)) + sigma0 * Z  ,  Z ~ N(0,1)

    Returns: (N, T) 배열
    """
    np.random.seed(seed)
    kappa  = base_params["kappa"]
    mu     = base_params["mu"]
    sigma0 = sigma0_override          # 시나리오별 sigma 주입

    paths = np.zeros((N, T + 1))
    paths[:, 0] = x0
    Z = np.random.standard_normal((N, T))

    for t in range(T):
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (mu - paths[:, t])
            + sigma0 * Z[:, t]
        )
    return paths[:, 1:]   # (N, T)


# ──────────────────────────────────────────────────────────────
# 6. ETF 가격경로 조합
# ──────────────────────────────────────────────────────────────

def combine_etf_paths(
    gap_paths: np.ndarray,   # (N, T)
    kp_paths: np.ndarray,    # (N, T)
    S0: float,
    g0: float,
    kp0: float,
) -> np.ndarray:
    """ETF_t = S0 * (1 + GAP_t) * (1 + KP_t) / ((1 + g0) * (1 + kp0)).

    초기값 정규화: t=0 시점의 GAP=g0, KP=kp0 에서 ETF=S0.
    """
    denom = (1 + g0) * (1 + kp0) if (1 + g0) * (1 + kp0) != 0 else 1.0
    etf   = S0 * (1 + gap_paths) * (1 + kp_paths) / denom
    return etf   # (N, T)


# ──────────────────────────────────────────────────────────────
# 7. 리스크 지표
# ──────────────────────────────────────────────────────────────

def compute_risk_metrics(
    scenario_id: str,
    sc_type: str,
    combo: str,
    sigma_gap_ann: float,
    sigma_kp_ann: float,
    terminal: np.ndarray,
    S0: float,
) -> dict:
    var95  = np.percentile(terminal, 5)
    var99  = np.percentile(terminal, 1)
    tail95 = terminal[terminal <= var95]
    tail99 = terminal[terminal <= var99]
    return {
        "Scenario_ID"   : scenario_id,
        "Type"          : sc_type,
        "combo_label"   : combo,
        "σ_GAP(ann)"    : round(sigma_gap_ann, 4),
        "σ_KP(ann)"     : round(sigma_kp_ann,  4),
        "E[ETF_T]"      : round(terminal.mean(), 4),
        "Median"        : round(np.median(terminal), 4),
        "VaR95%"        : round(var95, 4),
        "VaR99%"        : round(var99, 4),
        "CVaR95%"       : round(tail95.mean() if len(tail95) > 0 else var95, 4),
        "CVaR99%"       : round(tail99.mean() if len(tail99) > 0 else var99, 4),
        "P(loss>20%)"   : round(np.mean(terminal < S0 * 0.80), 4),
    }


# ──────────────────────────────────────────────────────────────
# 8. 시각화
# ──────────────────────────────────────────────────────────────

def _pct(mat: np.ndarray, q: float) -> np.ndarray:
    return np.percentile(mat, q, axis=0)


def plot_scenario_paths(
    results: List[dict],
    S0: float,
    T: int,
    out_dir: Path,
) -> None:
    """시나리오별 ETF 가격경로 + 말기 분포 멀티패널 시각화."""
    n = len(results)
    if n == 0:
        return

    days = np.arange(1, T + 1)
    ncols = min(n, 4)
    nrows_path = (n + ncols - 1) // ncols
    nrows_dist = (n + ncols - 1) // ncols

    fig_h = 5 * (nrows_path + nrows_dist) + 5
    fig   = plt.figure(figsize=(5 * ncols, fig_h))
    gs    = fig.add_gridspec(
        nrows_path + nrows_dist + 1, ncols,
        hspace=0.5, wspace=0.3,
    )

    # ── 시나리오별 경로 + 분포 ──────────────────────────────
    for idx, res in enumerate(results):
        row_path = idx // ncols
        row_dist = nrows_path + idx // ncols
        col      = idx % ncols
        etf      = res["etf_paths"]     # (N, T)
        terminal = etf[:, -1]
        color    = _TYPE_COLORS.get(res["type"], "gray")
        title    = f"{res['scenario_id']}\n{res['combo_label']}"

        # 경로 팬
        ax_p = fig.add_subplot(gs[row_path, col])
        for i in range(min(200, len(etf))):
            ax_p.plot(days, etf[i], color=color, alpha=0.04, linewidth=0.4)
        for q, ls, lw, lbl in [(5, "--", 1.0, "P5/P95"), (50, "-", 2.0, "Median"), (95, "--", 1.0, None)]:
            ax_p.plot(days, _pct(etf, q), color=color, linestyle=ls, linewidth=lw, label=lbl)
        ax_p.axhline(S0, color="black", linewidth=0.7, linestyle=":")
        ax_p.set_title(f"{title}\nσ_GAP={res['sigma_gap_ann']:.4f}  σ_KP={res['sigma_kp_ann']:.4f}",
                       fontsize=8, fontweight="bold")
        ax_p.set_xlabel("Trading Days", fontsize=7)
        ax_p.set_ylabel("ETF Price", fontsize=7)
        ax_p.tick_params(labelsize=7)
        ax_p.legend(fontsize=7, loc="upper left")
        ax_p.grid(True, alpha=0.2)

        # 말기 분포
        ax_d = fig.add_subplot(gs[row_dist, col])
        ax_d.hist(terminal, bins=60, density=True, color=color, alpha=0.5, edgecolor="none")
        try:
            shape, loc, scale = scipy_stats.lognorm.fit(terminal[terminal > 0], floc=0)
            xr = np.linspace(terminal.min(), terminal.max(), 300)
            ax_d.plot(xr, scipy_stats.lognorm.pdf(xr, shape, loc, scale), "k-", linewidth=1.2)
        except Exception:
            pass
        var95 = np.percentile(terminal, 5)
        var99 = np.percentile(terminal, 1)
        ax_d.axvline(var95, color="orange", linewidth=1.2, linestyle="--", label=f"VaR95%={var95:.1f}")
        ax_d.axvline(var99, color="red",    linewidth=1.2, linestyle="--", label=f"VaR99%={var99:.1f}")
        ax_d.set_title(f"{title} — Terminal Dist.", fontsize=8, fontweight="bold")
        ax_d.set_xlabel("ETF S(T)", fontsize=7)
        ax_d.set_ylabel("Density", fontsize=7)
        ax_d.tick_params(labelsize=7)
        ax_d.legend(fontsize=7)
        ax_d.grid(True, alpha=0.2)

    # ── 시나리오 비교 패널 (최하단 행 전체 폭) ────────────────
    ax_comp = fig.add_subplot(gs[nrows_path + nrows_dist, :])
    for res in results:
        etf   = res["etf_paths"]
        color = _TYPE_COLORS.get(res["type"], "gray")
        lbl   = (f"{res['scenario_id']} [{res['combo_label'][:20]}] "
                 f"σ_G={res['sigma_gap_ann']:.3f} σ_K={res['sigma_kp_ann']:.3f}")
        ax_comp.plot(days, _pct(etf, 50), color=color, linewidth=1.5, label=lbl)
        ax_comp.fill_between(days, _pct(etf, 5), _pct(etf, 95), color=color, alpha=0.08)

    ax_comp.axhline(S0, color="black", linewidth=0.7, linestyle=":")
    ax_comp.set_title("Scenario Comparison — Median ETF Path + 90% Band",
                      fontsize=10, fontweight="bold")
    ax_comp.set_xlabel("Trading Days", fontsize=9)
    ax_comp.set_ylabel("ETF Price", fontsize=9)
    ax_comp.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_comp.grid(True, alpha=0.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario_regime_etf_paths.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[저장] {out_path}")


# ──────────────────────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    S0 = 10_000.0  # 초기 ETF 가격 (10,000원 앵커)
    T  = 252        # 시뮬레이션 기간 (거래일)
    N  = 5_000      # 몬테카를로 경로 수

    # 1) 시나리오 테이블 로드
    scenarios_df = load_scenarios()

    # 2) 일별 5변수 레짐 레이블 구축 (HMM 파이프라인)
    df_labeled = build_daily_labeled_df(n_init=10, B=500)

    # 3) GAP / KP 원시 시계열 로드
    print("\n[Step 2] y_variables.csv 로드...")
    y_vars = pd.read_csv(
        _ROOT / "dataset" / "raw" / "y_variables.csv",
        parse_dates=["Date"],
    ).set_index("Date").sort_index()
    gap_series = y_vars["etf_premium"].dropna()
    kp_series  = y_vars["Kimchi Premium"].dropna()
    print(f"    GAP 기간: {gap_series.index[0].date()} ~ {gap_series.index[-1].date()}  (n={len(gap_series)})")
    print(f"    KP  기간: {kp_series.index[0].date()} ~ {kp_series.index[-1].date()}   (n={len(kp_series)})")

    # 4) 시나리오별 sigma 추정
    print("\n[Step 3] 시나리오별 GAP/KP sigma 추정...")
    sigma_df = estimate_sigma_per_scenario(
        scenarios_df, df_labeled, gap_series, kp_series,
    )

    # 5) 전체 표본 OU 기본 파라미터 적합
    print("\n[Step 4] 전체 표본 OU 파라미터 적합...")
    gap_base_ou, kp_base_ou = fit_base_ou_params(gap_series, kp_series)
    g0  = float(gap_series.iloc[0])
    kp0 = float(kp_series.iloc[0])

    # 6) 시나리오별 시뮬레이션
    print(f"\n[Step 5] 시나리오별 시뮬레이션 (N={N}, T={T})...")
    sim_results  = []
    metrics_rows = []

    for sc_idx, (_, sc) in enumerate(sigma_df.iterrows()):
        sid  = sc["Scenario_ID"]
        sc_type = sc["Type"]
        combo   = sc["combo_label"]
        sig_gap = sc["sigma_gap_daily"]
        sig_kp  = sc["sigma_kp_daily"]
        sig_gap_ann = sc["sigma_gap_ann"]
        sig_kp_ann  = sc["sigma_kp_ann"]

        print(f"    {sid} [{combo}]  σ_GAP={sig_gap_ann:.4f}  σ_KP={sig_kp_ann:.4f}")

        # 시나리오마다 다른 seed → sigma가 같아도 경로 형태가 다름
        gap_paths = simulate_ou_paths(
            gap_base_ou, sig_gap, g0, T, N, seed=42  + sc_idx * 1000,
        )
        kp_paths  = simulate_ou_paths(
            kp_base_ou,  sig_kp,  kp0, T, N, seed=123 + sc_idx * 1000,
        )
        etf_paths = combine_etf_paths(gap_paths, kp_paths, S0, g0, kp0)

        sim_results.append({
            "scenario_id"  : sid,
            "type"         : sc_type,
            "combo_label"  : combo,
            "sigma_gap_ann": sig_gap_ann,
            "sigma_kp_ann" : sig_kp_ann,
            "etf_paths"    : etf_paths,
            "gap_paths"    : gap_paths,
            "kp_paths"     : kp_paths,
        })

        metrics_rows.append(compute_risk_metrics(
            sid, sc_type, combo,
            sig_gap_ann, sig_kp_ann,
            etf_paths[:, -1], S0,
        ))

    # 7) 리스크 지표 출력 및 저장
    risk_df = pd.DataFrame(metrics_rows)
    print("\n=== Risk Metrics Table ===")
    print(risk_df.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    risk_csv = OUT_DIR / "scenario_regime_risk_metrics.csv"
    risk_df.to_csv(risk_csv, index=False, encoding="utf-8-sig")
    sigma_csv = OUT_DIR / "scenario_regime_sigma.csv"
    sigma_df.to_csv(sigma_csv, index=False, encoding="utf-8-sig")
    print(f"\n[저장] {risk_csv}")
    print(f"[저장] {sigma_csv}")

    # 8) 시각화
    print("\n[Step 6] 시각화 생성...")
    plot_scenario_paths(sim_results, S0=S0, T=T, out_dir=OUT_DIR)

    print("\n완료.")
    return risk_df


if __name__ == "__main__":
    main()
