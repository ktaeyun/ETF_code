"""
simulator/results_main.py
==========================
Base 대비 시나리오 리스크 지표 통계적 유의성 검정

배경
----
scenario_main.py 는 시나리오(Base, S01~S09)마다 몬테카를로(N=1000) 시뮬레이션을
"1회" 실행해 VaR_95 / CVaR_95 / VaR_99 / CVaR_99 / Volatility / Max_DD_mean /
p50 / p95 등 리스크 지표의 점추정치만 산출한다(korean_etf_risk_metrics.csv).
(VaR_95와 p5는 종단가격 분포 하위 5% 분위수로 동일 지표이므로 p5는 제외한다.)
이 스크립트는 그 점추정치가 시뮬레이션 노이즈 범위 안의 우연인지, Base 대비
통계적으로 유의미한 차이(우연/표본오차 여부)인지만 검정한다 — 실무적 중요도
(economic significance) 판단은 본 검정의 범위에 포함하지 않는다.
scenario_main.py / main.py 의 기존 로직은 수정하지 않고 그 안의 함수를 그대로
재사용한다.

절차
----
1. 반복 시뮬레이션 설계: Base/시나리오 각각에 대해 서로 다른 random seed로
   N=1000 경로 시뮬레이션을 M=100회 독립 반복한다.
   - 반복 m마다 Base와 모든 시나리오가 동일한 시드(Common Random Numbers)를
     사용하도록 하여, m번째 반복의 base_m과 scenario_m을 짝짓는다.
   - GAP/KP 모수(kappa, mu, sigma0, delta...)는 Base·시나리오 공통으로 고정
     (results/simulator/cache/sim_meta.json). 시나리오 간 차이는 오직
     외생변수 시계열(si/vix/vol/kospi)에서만 발생 — run_single_scenario와 동일.
   - NAV 경로는 시나리오와 무관하므로 results/simulator/cache/sim_arrays.npz
     캐시(고정 1000경로)를 모든 반복에서 그대로 재사용한다.
2. 지표 계산 (scenario_main.compute_risk_metrics 재사용)
   - VaR_95/CVaR_95/VaR_99/CVaR_99/p50/p95: 각 반복의 N=1000개 경로 중
     t=T 시점(terminal value)의 분포에서 계산.
   - Volatility/Max_DD_mean: 경로 전체(t=1~T)를 사용해 계산.
3. 반복별 차이 계산: 각 반복 m마다 diff_m = Scenario_m - Base_m 을 지표별로
   계산한다 (m=1,...,M).
4. 정규성 사전 검정: 지표별로 diff_m(M개)의 정규성을 Shapiro-Wilk로 확인한다.
   p_shapiro > 0.05 → 정규성 만족, p_shapiro <= 0.05 → 정규성 위반.
5. 검정 방법 분기
   - 정규성 만족 시: 대응표본 t-검정(Paired t-test).
     diff_mean = diff의 평균, se = SD(diff)/sqrt(M), t = diff_mean/se,
     자유도 M-1인 t분포로 p-value 산출.
   - 정규성 위반 시: Wilcoxon 부호순위검정(Wilcoxon Signed-Rank Test).
     |diff_m| 기준 순위를 매기고 부호를 부여해 W+/W-를 구한 뒤, M이 크므로
     정규근사 z = (W-mu_W)/sigma_W (mu_W=M(M+1)/4, sigma_W=sqrt(M(M+1)(2M+1)/24))
     로 양측 p-value 산출 (scipy.stats.wilcoxon(mode="approx")로 구현, 수식
     일치 검증 완료).
6. 판정: p < 0.05 → 통계적으로 유의미한 차이(우연이 아님), p >= 0.05 → 유의미한
   차이 확인되지 않음.
7. 방향 제시: 유의미한 지표는 diff_mean의 부호로 증가(increase)/감소(decrease)
   방향을 함께 제시한다.
8. 다중비교 보정 미적용: 9개 시나리오 x 8개 지표(VaR_95=p5 통합 후)의 모든
   비교를 Bonferroni 등 보정 없이 그대로 보고한다 (Rothman, 1990 근거 — 사전에
   경제적 의미를 갖고 설계된 시나리오이므로 "만능귀무가설" 전제가 부적절).

실행 예시
---------
  python simulator/results_main.py                       (M=100, N=1000, 전체 시나리오)
  python simulator/results_main.py --M 20 --N 200         (빠른 테스트)
  python simulator/results_main.py --scenarios S01,S03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.scenario_generator import (
    generate_scenario_series,
    load_hmm_results_cache,
    save_hmm_results_cache,
    scenarios_from_csv,
)
from simulator.data_loader import load_gap_exog, load_kp_exog
from simulator.gap_ou_simulator import GapOUSimulator
from simulator.kp_threshold_ou_simulator import KPThresholdOUSimulator
from simulator.scenario_main import (
    SCENARIO_CSV,
    OUT_BASE,
    _build_hmm_with_fixed_k,
    _build_korean_etf_paths,
    compute_risk_metrics,
    load_base_mc_arrays,
)

_BASE_DIR = _ROOT / "results" / "simulator"
_SIM_META_PATH = _BASE_DIR / "cache" / "sim_meta.json"

# 배경 절에서 지정한 리스크 지표 (compute_risk_metrics 출력 키와 동일).
# VaR_95와 p5는 동일 지표(종단가격 분포 하위 5% 분위수)이므로 중복 계산하지 않고
# VaR_95 하나로 통일한다 (p5 제외).
METRICS = ["VaR_95", "CVaR_95", "VaR_99", "CVaR_99",
           "Volatility", "Max_DD_mean", "p50", "p95"]

# ──────────────────────────────────────────────────────────────
# 측정 기준(basis)
#
# 이 연구의 시나리오는 GAP과 KP만 조작하고 NAV는 Base로 고정한다. 따라서 전체
# ETF 가격에서 지표를 재면 조작하지 않은 NAV의 변동이 분모에 함께 들어가
# 시나리오 효과가 희석된다. 실제로 최종시점 로그가격 분산의 100%를 NAV가
# 차지하고 GAP+KP는 0.02%에 그친다(일별 변화 기준으로는 GAP+KP가 11.6%).
#
# 그래서 두 기준으로 나란히 산출한다.
#   etf : NAV x (1+GAP) x (1+KP)  — 투자자가 실제로 보는 가격. 기존 기준
#   kr  : (1+GAP) x (1+KP)        — 조작 대상만 분리. 시나리오 효과를 직접 측정
# ──────────────────────────────────────────────────────────────
BASES = {
    "etf": {"label": "전체 ETF (NAV x (1+GAP) x (1+KP))", "include_nav": True},
    "kr":  {"label": "한국 요인 ((1+GAP) x (1+KP))",       "include_nav": False},
}


def _build_paths(mc_nav: np.ndarray, mc_gap: np.ndarray, mc_kp: np.ndarray,
                 anchor: float, include_nav: bool) -> np.ndarray:
    """기준에 따라 경로를 만든 뒤 anchor(원)로 정규화.

    include_nav=True 면 기존 _build_korean_etf_paths 와 동일한 결과를 낸다.
    """
    if include_nav:
        return _build_korean_etf_paths(mc_nav, mc_gap, mc_kp, anchor)

    min_N = min(mc_gap.shape[0], mc_kp.shape[0])
    min_T = min(mc_gap.shape[1], mc_kp.shape[1])
    raw = (1 + mc_gap[:min_N, :min_T]) * (1 + mc_kp[:min_N, :min_T])
    init = raw[:, 0:1]
    init = np.where(init == 0, 1.0, init)
    return raw * (anchor / init)

# ══════════════════════════════════════════════════════════════
# 1. Base/시나리오 공통 파라미터 + 외생변수 준비
# ══════════════════════════════════════════════════════════════

def _load_fixed_gap_kp_simulators() -> tuple[GapOUSimulator, KPThresholdOUSimulator]:
    """sim_meta.json 캐시에서 GAP/KP 모수를 로드.

    scenario_main.run_single_scenario 의 3~4단계("Base 모수 로드(고정)")와
    동일한 구성 — Base 와 모든 시나리오가 공유하는 고정 모수다.
    """
    if not _SIM_META_PATH.exists():
        raise FileNotFoundError(
            f"Base 시뮬레이터 캐시 없음: {_SIM_META_PATH}\n"
            "simulator/main.py 를 먼저 실행하세요."
        )
    with open(_SIM_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    gap_p = meta["gap_params"]
    kp_p = meta["kp_params"]

    df_gap_hist = load_gap_exog(base_dir=str(_ROOT))
    df_kp_hist = load_kp_exog(base_dir=str(_ROOT))

    gap_sim = GapOUSimulator(
        kappa=gap_p["kappa"], mu=gap_p["mu"], sigma0=gap_p["sigma0"],
        delta1=gap_p["delta1"], delta2=gap_p["delta2"],
        si_mean=float(np.mean(df_gap_hist["value"].values)),
        si_std=float(np.std(df_gap_hist["value"].values)),
        vix_mean=float(np.mean(df_gap_hist["btc_volatility"].values)),
        vix_std=float(np.std(df_gap_hist["btc_volatility"].values)),
        clip=3.0,
        nu=gap_p.get("nu"),
    )

    kp_rp = kp_p["regime_params"]
    _log_vol_raw = np.log(df_kp_hist["volume_btc"].values + 1e-8)
    kp_sim = KPThresholdOUSimulator({
        "threshold": kp_p["threshold"],
        "regime_params": {
            int(r): {"kappa": v["kappa"], "mu": v["mu"], "sigma0": v["sigma0"]}
            for r, v in kp_rp.items()
        },
        "delta1_regime": {int(r): v["delta1"] for r, v in kp_rp.items()},
        "delta2_regime": {int(r): v["delta2"] for r, v in kp_rp.items()},
        "delta3_regime": {int(r): v["delta3"] for r, v in kp_rp.items()},
        "vol_btc_mean": float(np.mean(_log_vol_raw)),
        "vol_btc_std": float(np.std(_log_vol_raw)),
        "kospi_mean": float(np.mean(df_kp_hist["KOSPI_Volatility"].values)),
        "kospi_std": float(np.std(df_kp_hist["KOSPI_Volatility"].values)),
        "bitcoin_kr_mean": float(np.mean(df_kp_hist["bitcoin_kr"].values)),
        "bitcoin_kr_std": float(np.std(df_kp_hist["bitcoin_kr"].values)),
    }, clip=3.0)

    return gap_sim, kp_sim


def _align(sc_arr: np.ndarray, hist_len: int) -> np.ndarray:
    if len(sc_arr) >= hist_len:
        return sc_arr[:hist_len]
    return np.pad(sc_arr, (0, hist_len - len(sc_arr)), mode="edge")


def _prepare_series(
    scenario: dict | None,
    hmm_results: dict,
    df_gap_hist: pd.DataFrame,
    df_kp_hist: pd.DataFrame,
    T_gen: int,
    seed: int,
) -> dict:
    """Base(scenario=None)는 실제 외생변수, 시나리오는 레짐 기반 생성 외생변수를 사용.

    run_single_scenario 의 1~2단계와 동일한 구성.
    """
    gap_series = df_gap_hist["etf_premium"]
    kp_series = df_kp_hist["Kimchi Premium"]
    T_gap, T_kp = len(gap_series), len(kp_series)

    if scenario is None:  # Base: 실제 외생변수 그대로 사용
        si_arr = df_gap_hist["value"].values
        vix_arr = df_gap_hist["btc_volatility"].values
        vol_arr = df_kp_hist["volume_btc"].values
        kv_arr = df_kp_hist["KOSPI_Volatility"].values
        bkr_arr = df_kp_hist["bitcoin_kr"].values
    else:  # 시나리오: 레짐 기반 생성 외생변수로 교체
        df_gap_sc = generate_scenario_series(hmm_results, scenario, T=T_gen, seed=seed, inverse=True)
        df_kp_sc = generate_scenario_series(hmm_results, scenario, T=T_gen, seed=seed, inverse=True)
        si_arr = _align(df_gap_sc["global_btc_svi"].values, T_gap)
        vix_arr = _align(df_gap_sc["Global_RV"].values, T_gap)
        vol_arr = _align(df_kp_sc["btc_volume_btc"].values, T_kp)
        kv_arr = _align(df_kp_sc["VKOSPI_resid"].values, T_kp)
        # domestic_btc_svi(= bitcoin_kr, KR_SVI)도 시나리오 정의 변수이므로 함께 교체한다
        bkr_arr = _align(df_kp_sc["domestic_btc_svi"].values, T_kp)

    return {
        "actual_gap": np.asarray(gap_series).flatten(),
        "actual_kp": np.asarray(kp_series).flatten(),
        "T_gap": T_gap, "T_kp": T_kp,
        "si": si_arr, "vix": vix_arr, "vol": vol_arr, "kv": kv_arr,
        "bkr": bkr_arr,
    }


# ══════════════════════════════════════════════════════════════
# 2. 반복(m)별 몬테카를로 배치 + 리스크 지표
# ══════════════════════════════════════════════════════════════

def _run_mc_batch(
    gap_sim: GapOUSimulator, kp_sim: KPThresholdOUSimulator,
    series: dict, N: int, seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """N개 경로를 한 배치로 시뮬레이션. seed_base 가 같으면(=CRN) Base/시나리오가
    동일한 epsilon_t 난수열을 사용하게 된다 (gap_ou_simulator.simulate_gap,
    kp_threshold_ou_simulator.simulate_kp 는 매 호출마다 np.random.seed(seed) 후
    순차적으로 표준정규 난수를 뽑으므로, T가 같으면 seed가 같을 때 난수열도 같다).
    """
    mc_gap = np.array([
        np.asarray(gap_sim.simulate_gap(
            T=series["T_gap"], g0=series["actual_gap"][0],
            si_future=series["si"], vix_future=series["vix"],
            seed=seed_base + 10_000 + i,
        )).flatten()
        for i in range(N)
    ])
    mc_kp = np.array([
        np.asarray(kp_sim.simulate_kp(
            T=series["T_kp"], kp0=series["actual_kp"][0],
            volume_btc_future=series["vol"], kospi_vol_future=series["kv"],
            bitcoin_kr_future=series["bkr"],
            seed=seed_base + 20_000 + i,
        )).flatten()
        for i in range(N)
    ])
    return mc_gap, mc_kp


def _load_actual_etf(min_T: int, anchor: float) -> np.ndarray:
    etf_csv = _BASE_DIR / "korean_etf_price.csv"
    if etf_csv.exists():
        arr = pd.read_csv(etf_csv)["actual_etf_krw"].values
        if len(arr) >= min_T:
            return arr
    return np.full(min_T, anchor)


def _actual_korean_factor(series: dict, min_T: int, anchor: float) -> np.ndarray:
    """한국 요인 기준의 실제 대응물: (1+실제GAP)(1+실제KP)를 anchor로 정규화.

    compute_risk_metrics 는 이 값을 MAE_actual 계산에만 쓴다. 기준이 다르면
    비교 대상도 달라져야 하므로 전체 ETF 가격을 그대로 넘기지 않는다.
    """
    g = np.asarray(series["actual_gap"]).flatten()[:min_T]
    k = np.asarray(series["actual_kp"]).flatten()[:min_T]
    n = min(len(g), len(k), min_T)
    raw = (1 + g[:n]) * (1 + k[:n])
    if n < min_T:
        raw = np.pad(raw, (0, min_T - n), mode="edge")
    init = raw[0] if raw[0] != 0 else 1.0
    return raw * (anchor / init)


def run_repeated_mc(
    scenario_id: str,
    scenario: dict | None,
    hmm_results: dict,
    gap_sim: GapOUSimulator,
    kp_sim: KPThresholdOUSimulator,
    df_gap_hist: pd.DataFrame,
    df_kp_hist: pd.DataFrame,
    mc_nav: np.ndarray,
    M: int, N: int, T_gen: int, seed: int, anchor: float = 10000.0,
) -> dict[str, pd.DataFrame]:
    """scenario_id 에 대해 M회 독립 반복(CRN)으로 N개 경로씩 시뮬레이션하고,
    반복마다 각 기준(BASES)의 리스크 지표를 계산한다.

    Returns
    -------
    {basis_key: (M행 x 지표) DataFrame}

    두 기준은 동일한 mc_gap / mc_kp 를 공유하고 결합 방식만 다르므로,
    기준을 하나 더 늘려도 시뮬레이션 비용은 늘지 않는다.
    """
    series = _prepare_series(scenario, hmm_results, df_gap_hist, df_kp_hist, T_gen, seed)

    rows = {b: [] for b in BASES}
    for m in range(M):
        seed_base_m = seed + m * 1_000_000  # Base/시나리오 공통 산식 → CRN
        mc_gap, mc_kp = _run_mc_batch(gap_sim, kp_sim, series, N, seed_base_m)

        for b, cfg in BASES.items():
            paths = _build_paths(mc_nav, mc_gap, mc_kp, anchor, cfg["include_nav"])
            T_b = paths.shape[1]
            actual = (_load_actual_etf(T_b, anchor) if cfg["include_nav"]
                      else _actual_korean_factor(series, T_b, anchor))
            risk = compute_risk_metrics(paths, actual, label=scenario_id)
            risk["m"] = m
            rows[b].append(risk)

        if (m + 1) % 10 == 0 or m == M - 1:
            print(f"    [{scenario_id}] 반복 {m + 1}/{M} 완료")

    return {b: pd.DataFrame(v) for b, v in rows.items()}


# ══════════════════════════════════════════════════════════════
# 3. Base vs 시나리오 통계적 유의성 검정
# ══════════════════════════════════════════════════════════════

def paired_t_test(base_vals: np.ndarray, sc_vals: np.ndarray, alpha: float = 0.05) -> dict:
    """대응표본 t-검정 (Paired t-test) — 정규성 만족 시 사용.

    CRN으로 짝지어진 M개 반복 base_1..M, scenario_1..M에서 diff_m = scenario_m - base_m
    을 계산하고, diff_mean/SE(=SD(diff)/sqrt(M))로 t통계량을 구해 자유도 M-1인
    t분포로 p-value를 산출한다. p < alpha 이면 유의미한 차이로 판정한다.
    """
    diff = sc_vals - base_vals
    n = len(diff)
    diff_mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    if not np.isfinite(se) or se <= 0:
        t_stat, pvalue = np.nan, np.nan
    else:
        t_stat = diff_mean / se
        pvalue = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1)))
    significant = bool(pvalue < alpha) if not np.isnan(pvalue) else False
    return {
        "method": "paired_t_test",
        "diff_mean": diff_mean,
        "statistic": t_stat,
        "pvalue": pvalue,
        "significant": significant,
    }


def wilcoxon_signed_rank_test(base_vals: np.ndarray, sc_vals: np.ndarray, alpha: float = 0.05) -> dict:
    """Wilcoxon 부호순위검정 — 정규성 위반 시 사용 (M이 크므로 정규근사).

    |diff_m| 기준 순위를 매기고 부호를 부여해 W+(양의 순위합), W-(음의 순위합)를
    구한 뒤, mu_W = M(M+1)/4, sigma_W = sqrt(M(M+1)(2M+1)/24)로 정규근사
    z = (W - mu_W)/sigma_W, 양측 p-value = P(|Z| >= |z|)를 산출한다.
    scipy.stats.wilcoxon(mode="approx", correction=False)가 이 정규근사 공식과
    동일한 p-value를 준다 (수식 검증 완료).
    """
    diff = sc_vals - base_vals
    diff_mean = float(diff.mean())
    try:
        stat, pvalue = sp_stats.wilcoxon(diff, zero_method="wilcox", correction=False, mode="approx")
        stat, pvalue = float(stat), float(pvalue)
    except ValueError:
        # 모든 diff_m이 0인 경우 등 극단적 케이스 (연속값 시뮬레이션에서는 사실상 발생하지 않음)
        stat, pvalue = np.nan, np.nan
    significant = bool(pvalue < alpha) if not np.isnan(pvalue) else False
    return {
        "method": "wilcoxon",
        "diff_mean": diff_mean,
        "statistic": stat,
        "pvalue": pvalue,
        "significant": significant,
    }


def compare_base_vs_scenario(
    base_df: pd.DataFrame, sc_df: pd.DataFrame, scenario_id: str, alpha: float = 0.05,
) -> list[dict]:
    """Base(base_df) 대비 시나리오(sc_df)의 지표별 유의성 검정.

    지표별로 diff_m(M개)의 정규성을 Shapiro-Wilk로 먼저 확인한 뒤,
    정규성 만족 시 Paired t-test, 위반 시 Wilcoxon 부호순위검정을 적용한다.
    유의성 판정(주 결론)과 pct_change(Base 대비 % 변화, 보조 캐비엇)를 나란히
    보고한다 — "통계적으로는 유의미하나 경제적 크기는 작을 수 있다"는 점을
    드러내기 위함. 실무적 중요도(effect size) 판단은 본 검정의 범위 밖이다.
    """
    rows = []
    for metric in METRICS:
        base_vals = base_df[metric].values.astype(float)
        sc_vals = sc_df[metric].values.astype(float)
        diff = sc_vals - base_vals

        if len(diff) >= 3:
            _, shapiro_p = sp_stats.shapiro(diff)
            shapiro_p = float(shapiro_p)
        else:
            shapiro_p = np.nan
        normal = bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else False

        test = paired_t_test(base_vals, sc_vals, alpha=alpha) if normal \
            else wilcoxon_signed_rank_test(base_vals, sc_vals, alpha=alpha)

        base_mean = float(base_vals.mean())
        pct_change = (test["diff_mean"] / abs(base_mean) * 100) if abs(base_mean) > 1e-12 else np.nan
        if test["diff_mean"] > 0:
            direction = "increase"
        elif test["diff_mean"] < 0:
            direction = "decrease"
        else:
            direction = "unchanged"

        rows.append({
            "scenario_id": scenario_id,
            "metric": metric,
            "base_mean": base_mean,
            "scenario_mean": float(sc_vals.mean()),
            "diff_mean": test["diff_mean"],
            "pct_change": pct_change,
            "shapiro_p": shapiro_p,
            "method": test["method"],
            "statistic": test["statistic"],
            "pvalue": test["pvalue"],
            "significant": test["significant"],
            "direction": direction,
        })
    return rows


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Base 대비 시나리오 리스크 지표 통계적 유의성 검정")
    parser.add_argument("--scenario-csv", type=str, default=str(SCENARIO_CSV))
    parser.add_argument("--out-dir", type=str, default=str(OUT_BASE / "significance_test"))
    parser.add_argument("--T", type=int, default=252, help="시나리오 외생변수 생성 길이")
    parser.add_argument("--N", type=int, default=1000, help="반복(m)당 몬테카를로 경로 수")
    parser.add_argument("--M", type=int, default=100, help="독립 반복 횟수 (CRN 적용)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05, help="유의수준")
    parser.add_argument("--scenarios", type=str, default=None, help="쉼표구분 시나리오 ID (기본: 전체)")
    parser.add_argument("--force-refit", action="store_true", help="HMM 캐시 무시하고 재실행")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: HMM 파라미터 로드 (scenario_main.py Step1과 동일 패턴) ──────
    print("=" * 60)
    print("  [Step 1] HMM 파라미터 로드")
    print("=" * 60)
    _hmm_cache_dir = _ROOT / "results" / "cache"
    hmm_results = None
    if not args.force_refit:
        hmm_results = load_hmm_results_cache(_hmm_cache_dir, n_init=10, B=1000)
    if hmm_results is None:
        _regime_cache = _hmm_cache_dir / "regime_df_cache.csv"
        if not _regime_cache.exists():
            raise FileNotFoundError(
                f"레짐 캐시 없음: {_regime_cache}\n"
                "simulator/scenario_main.py 를 먼저 한 번 실행해 캐시를 생성하세요."
            )
        hmm_results = _build_hmm_with_fixed_k(_regime_cache)
        save_hmm_results_cache(hmm_results, _hmm_cache_dir, n_init=10, B=1000)

    # ── Step 2: 시나리오 로드 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 2] 시나리오 로드")
    print("=" * 60)
    scenario_csv = Path(args.scenario_csv)
    if not scenario_csv.exists():
        raise FileNotFoundError(f"시나리오 CSV 없음: {scenario_csv}")
    all_scenarios = scenarios_from_csv(scenario_csv, hmm_results)
    if args.scenarios:
        selected = set(args.scenarios.split(","))
        all_scenarios = {k: v for k, v in all_scenarios.items() if k in selected}
    print(f"  대상 시나리오: {list(all_scenarios.keys())}")

    # ── Step 3: 공통 리소스 준비 (모수/외생변수/Base NAV 캐시) ─────────────
    gap_sim, kp_sim = _load_fixed_gap_kp_simulators()
    df_gap_hist = load_gap_exog(base_dir=str(_ROOT))
    df_kp_hist = load_kp_exog(base_dir=str(_ROOT))

    base_mc = load_base_mc_arrays()
    if base_mc is None or "mc_nav" not in base_mc:
        raise FileNotFoundError(
            "Base MC 캐시(results/simulator/cache/sim_arrays.npz) 없음\n"
            "simulator/main.py 를 먼저 실행하세요."
        )
    mc_nav = base_mc["mc_nav"]

    est_calls = (len(all_scenarios) + 1) * args.M * args.N * 2
    print(f"\n  [예상 작업량] 총 시뮬레이션 호출 수 = {est_calls:,}회 "
          f"({len(all_scenarios) + 1}개 대상 x M={args.M} x N={args.N} x GAP/KP)")

    # ── Step 4: Base + 시나리오별 M회 독립 반복 ─────────────────────────
    print("\n" + "=" * 60)
    print(f"  [Step 4] Base + {len(all_scenarios)}개 시나리오, 각 M={args.M}회 반복 (N={args.N})")
    print("=" * 60)

    print(f"  측정 기준 {len(BASES)}종:")
    for b, cfg in BASES.items():
        print(f"    [{b}] {cfg['label']}")

    # per_basis[basis][scenario_id] = (M행 x 지표) DataFrame
    per_basis: dict[str, dict[str, pd.DataFrame]] = {b: {} for b in BASES}

    print("\n  -- Base --")
    _res = run_repeated_mc(
        "Base", None, hmm_results, gap_sim, kp_sim, df_gap_hist, df_kp_hist,
        mc_nav, M=args.M, N=args.N, T_gen=args.T, seed=args.seed,
    )
    for b in BASES:
        per_basis[b]["Base"] = _res[b]

    for sid, scenario in all_scenarios.items():
        print(f"\n  -- {sid} --")
        _res = run_repeated_mc(
            sid, scenario, hmm_results, gap_sim, kp_sim, df_gap_hist, df_kp_hist,
            mc_nav, M=args.M, N=args.N, T_gen=args.T, seed=args.seed,
        )
        for b in BASES:
            per_basis[b][sid] = _res[b]

    # ── Step 5: 기준별 저장 + Base 대비 유의성 검정 ─────────────────────
    print("\n" + "=" * 60)
    print("  [Step 5] Base 대비 통계적 유의성 검정 (다중비교 보정 미적용)")
    print("=" * 60)

    sig_by_basis: dict[str, pd.DataFrame] = {}
    eff_by_basis: dict[str, pd.DataFrame] = {}
    flag_by_basis: dict[str, pd.DataFrame] = {}

    for b, cfg in BASES.items():
        repeat_dfs = per_basis[b]

        # 반복별 리스크 지표 원본 (감사/재현용)
        raw_rows = []
        for sid, df in repeat_dfs.items():
            d = df.copy()
            d.insert(0, "scenario_id", sid)
            d.insert(1, "basis", b)
            raw_rows.append(d)
        pd.concat(raw_rows, ignore_index=True).to_csv(
            out_dir / f"raw_repeat_metrics_{b}.csv", index=False, encoding="utf-8-sig")

        base_df = repeat_dfs["Base"]
        sig_rows = []
        for sid in all_scenarios:
            sig_rows.extend(compare_base_vs_scenario(
                base_df, repeat_dfs[sid], sid, alpha=args.alpha
            ))
        sig_df = pd.DataFrame(sig_rows)
        sig_df.insert(0, "basis", b)
        sig_df.to_csv(out_dir / f"significance_results_{b}.csv",
                      index=False, encoding="utf-8-sig")

        pval = sig_df.pivot(index="metric", columns="scenario_id", values="pvalue")
        flag = sig_df.pivot(index="metric", columns="scenario_id", values="significant")
        eff = sig_df.pivot(index="metric", columns="scenario_id", values="pct_change")
        pval.to_csv(out_dir / f"significance_matrix_pvalue_{b}.csv", encoding="utf-8-sig")
        flag.to_csv(out_dir / f"significance_matrix_flag_{b}.csv", encoding="utf-8-sig")
        eff.to_csv(out_dir / f"effect_size_matrix_pct_{b}.csv", encoding="utf-8-sig")

        sig_by_basis[b], eff_by_basis[b], flag_by_basis[b] = sig_df, eff, flag

        n_sig = int(sig_df["significant"].sum())
        print(f"\n  [{b}] {cfg['label']}")
        print(f"      비교 {len(sig_df)}회 중 유의 {n_sig}회  ->  *_{b}.csv")

    # ── Step 6: 기준 간 효과크기 비교 ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 6] 기준 간 효과크기 비교")
    print("=" * 60)

    keys = list(BASES)
    n_sc = len(all_scenarios)
    cmp_rows = []
    for metric in METRICS:
        row = {"metric": metric}
        for b in keys:
            e = eff_by_basis[b].loc[metric]
            f = flag_by_basis[b].loc[metric]
            row[f"effect_median_{b}"] = float(e.abs().median())
            row[f"effect_S05_{b}"] = float(e.get("S05", float("nan")))
            row[f"n_significant_{b}"] = int(f.sum())
        denom = row[f"effect_median_{keys[0]}"]
        row["amplification"] = (row[f"effect_median_{keys[1]}"] / denom
                                if denom else float("nan"))
        cmp_rows.append(row)

    cmp_df = pd.DataFrame(cmp_rows).sort_values("effect_median_kr", ascending=False)
    cmp_df.to_csv(out_dir / "effect_size_comparison.csv",
                  index=False, encoding="utf-8-sig")

    hdr = "지표"
    print(f"\n  {hdr:<13}{'전체ETF 중앙':>14}{'한국요인 중앙':>15}{'증폭':>9}"
          f"{'전체 유의':>11}{'한국 유의':>11}")
    for _, r in cmp_df.iterrows():
        print(f"  {r['metric']:<13}{r['effect_median_etf']:13.3f}%{r['effect_median_kr']:14.3f}%"
              f"{r['amplification']:8.1f}x{int(r['n_significant_etf']):8d}/{n_sc}"
              f"{int(r['n_significant_kr']):8d}/{n_sc}")

    print(f"\n  S05(위기 시나리오) 효과크기")
    print(f"  {hdr:<13}{'전체ETF':>12}{'한국요인':>13}")
    for _, r in cmp_df.iterrows():
        print(f"  {r['metric']:<13}{r['effect_S05_etf']:+11.3f}%{r['effect_S05_kr']:+12.3f}%")

    print(f"\n  비교표 저장: {out_dir / 'effect_size_comparison.csv'}")
    print(f"  결과 디렉터리: {out_dir}")
    return {"per_basis": per_basis, "significance": sig_by_basis, "comparison": cmp_df}


if __name__ == "__main__":
    main()
