"""
regime_simulation.py
====================
HMM 기반 레짐 조건부 모수적 시뮬레이션
(Regime-Conditioned Parametric Simulation)

외생변수 매핑 (run_pipeline.py 기준):
  GAP 모델: global_btc_svi (SI),      Global_RV   (VIX proxy)
  KP  모델: btc_volume_btc (Volume),  VKOSPI      (KOSPI_Vol)

레짐 레이블:
  GAP 레짐: gaussian_hmm → Global_RV   HMM의 Viterbi 상태
  KP  레짐: gaussian_hmm → VKOSPI_resid HMM의 Viterbi 상태
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf
from statsmodels.tsa.stattools import adfuller

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════
# 상수 — 프로젝트 실제 변수명
# ══════════════════════════════════════════════════════════════

GAP_COLS: Tuple[str, str] = ("global_btc_svi", "Global_RV")   # (SI, VIX)
KP_COLS : Tuple[str, str] = ("btc_volume_btc", "VKOSPI")      # (Volume, KOSPI_Vol)

ALL_EXOG_COLS: List[str] = [
    "global_btc_svi", "Global_RV",
    "btc_volume_btc", "VKOSPI",
]


# ══════════════════════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════════════════════

@dataclass
class OUParams:
    """단일 변수의 OU 모수 추정 결과.

    OU 과정: X_t - X_{t-1} = θ(μ - X_{t-1})·Δt + σ·√Δt·ε_t
    OLS 회귀: y_t = α + β·x_t + η_t
      α = θ·μ·Δt,  β = -θ·Δt
      θ = -β/Δt,   μ = α/(-β),  σ = std(η)/√Δt

    Attributes
    ----------
    mu   : 장기 평균
    theta: 평균회귀 속도 (> 0 이면 안정)
    sigma: 확산 계수
    alpha: OLS 절편
    beta : OLS 기울기
    resid: OLS 잔차 η_t (정규화 전, shape=(n_pairs,))
    r2   : OLS 결정계수
    n_obs: 관측 쌍 수
    """

    mu   : float
    theta: float
    sigma: float
    alpha: float
    beta : float
    resid: np.ndarray
    r2   : float
    n_obs: int


@dataclass
class GAPSimParams:
    """GAP 과정 시뮬레이션 파라미터.

    g_t = g_{t-1} + κ·(μ_g - g_{t-1})·Δt + σ_t·√Δt·ε_gap
    σ_t = σ₀·exp(δ₁·SI_{t-1} + δ₂·VIX_{t-1})
    """

    kappa : float          # 평균회귀 속도
    mu    : float          # 장기 평균 GAP
    sigma0: float          # 기준 변동성
    delta1: float          # global_btc_svi (SI) 민감도
    delta2: float          # Global_RV (VIX) 민감도
    g0    : float = 0.0    # 초기 GAP 값


@dataclass
class KPSimParams:
    """KP 과정 시뮬레이션 파라미터.

    KP_t = KP_{t-1} + κ·(μ_kp - KP_{t-1})·Δt + σ_t·√Δt·ε_kp
    σ_t = σ₀·exp(δ₁·Vol_{t-1} + δ₂·KOSPI_{t-1})
    """

    kappa : float          # 평균회귀 속도
    mu    : float          # 장기 평균 KP
    sigma0: float          # 기준 변동성
    delta1: float          # btc_volume_btc (Volume) 민감도
    delta2: float          # VKOSPI (KOSPI_Vol) 민감도
    kp0   : float = 0.0    # 초기 KP 값


# ══════════════════════════════════════════════════════════════
# Task 1: 레짐별 내부 연속 쌍 추출
# ══════════════════════════════════════════════════════════════

def extract_consecutive_pairs(
    df_exog       : pd.DataFrame,
    regime_labels : pd.Series,
    columns       : List[str],
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """레짐 내부 연속 (t-1, t) 쌍 추출. 레짐 전환 경계는 제외.

    Parameters
    ----------
    df_exog      : 외생변수 DataFrame. DatetimeIndex 필요.
    regime_labels: HMM Viterbi 레짐 레이블 (0-indexed). df_exog 와 동일 인덱스.
    columns      : 처리할 컬럼명 리스트.
                   예: ["global_btc_svi", "Global_RV"]

    Returns
    -------
    dict : {regime: {col: [(X_{t-1}, X_t), ...]}}
           레짐 전환 경계면(t-1이 A레짐, t가 B레짐) 쌍은 제외됨.
    """
    common = df_exog.index.intersection(regime_labels.index)
    if len(common) == 0:
        raise ValueError("[extract_consecutive_pairs] 인덱스가 겹치지 않음.")

    missing = [c for c in columns if c not in df_exog.columns]
    if missing:
        raise ValueError(f"[extract_consecutive_pairs] 컬럼 없음: {missing}")

    df      = df_exog.loc[common, columns]
    labels  = regime_labels.loc[common].values
    regimes = sorted(set(labels.tolist()))

    result: Dict[int, Dict[str, List[Tuple[float, float]]]] = {
        int(r): {col: [] for col in columns} for r in regimes
    }

    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            continue                   # 레짐 전환 경계 제외
        r = int(labels[t])
        for col in columns:
            result[r][col].append((float(df.iloc[t - 1][col]),
                                   float(df.iloc[t][col])))

    for r in regimes:
        counts = {col: len(result[int(r)][col]) for col in columns}
        print(f"[Task1] regime={r}: {counts}")

    return result


# ══════════════════════════════════════════════════════════════
# Task 2: 레짐별 OU 모수 추정
# ══════════════════════════════════════════════════════════════

def estimate_ou_params(
    pairs_dict: Dict[int, Dict[str, List[Tuple[float, float]]]],
    dt        : float = 1 / 252,
) -> Dict[int, Dict[str, OUParams]]:
    """레짐별, 변수별 OU 모수를 OLS로 추정.

    OLS 설계:
      y_t = X_t - X_{t-1}
      x_t = X_{t-1}
      y_t = α + β·x_t + η_t
      θ = -β/Δt,  μ = α/(-β),  σ = std(η)/√Δt

    Parameters
    ----------
    pairs_dict: extract_consecutive_pairs() 의 출력.
    dt        : 시간 간격 Δt. Default=1/252 (일별).

    Returns
    -------
    dict : {regime: {col: OUParams}}
    """
    result: Dict[int, Dict[str, OUParams]] = {}

    for regime, col_pairs in pairs_dict.items():
        result[regime] = {}

        for col, pairs in col_pairs.items():
            if len(pairs) < 5:
                print(f"[Task2] regime={regime} {col}: 쌍 부족 ({len(pairs)}) — 건너뜀")
                continue

            arr = np.array(pairs)          # (n, 2)
            x   = arr[:, 0]               # X_{t-1}
            y   = arr[:, 1] - arr[:, 0]   # ΔX_t

            X_mat    = np.column_stack([np.ones(len(x)), x])
            coef, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            alpha, beta   = float(coef[0]), float(coef[1])

            y_hat = alpha + beta * x
            resid = y - y_hat

            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            theta = -beta / dt
            mu    = (alpha / (-beta)) if abs(beta) > 1e-12 else float(x.mean())
            sigma = float(np.std(resid, ddof=1)) / np.sqrt(dt)

            result[regime][col] = OUParams(
                mu=mu, theta=theta, sigma=sigma,
                alpha=alpha, beta=beta,
                resid=resid, r2=r2, n_obs=len(pairs),
            )

            print(
                f"[Task2] regime={regime} {col:20s}: "
                f"μ={mu:8.4f}  θ={theta:8.4f}  σ={sigma:8.4f}  R²={r2:.4f}"
            )

    return result


# ══════════════════════════════════════════════════════════════
# Task 3: 레짐별 변수 간 상관관계 추정
# ══════════════════════════════════════════════════════════════

def estimate_correlations(
    ou_params: Dict[int, Dict[str, OUParams]],
    groups   : Dict[str, Tuple[str, str]],
) -> Dict[int, Dict[str, float]]:
    """레짐별 잔차 간 Pearson 상관계수 추정.

    Parameters
    ----------
    ou_params: estimate_ou_params() 의 출력.
    groups   : {group_name: (col1, col2)} 상관 추정 쌍.
               예: {"GAP": ("global_btc_svi", "Global_RV"),
                    "KP":  ("btc_volume_btc", "VKOSPI")}

    Returns
    -------
    dict : {regime: {group_name: rho}}
           rho = Pearson 상관계수 (잔차 η_col1 vs η_col2)
    """
    result: Dict[int, Dict[str, float]] = {}

    for regime, col_params in ou_params.items():
        result[regime] = {}

        for group, (col1, col2) in groups.items():
            if col1 not in col_params or col2 not in col_params:
                print(f"[Task3] regime={regime} {group}: 모수 없음 — 0으로 설정")
                result[regime][group] = 0.0
                continue

            r1 = col_params[col1].resid
            r2 = col_params[col2].resid
            n  = min(len(r1), len(r2))

            if n < 3:
                result[regime][group] = 0.0
                continue

            rho, pval = pearsonr(r1[:n], r2[:n])
            result[regime][group] = float(rho)
            print(f"[Task3] regime={regime} {group}: ρ={rho:.4f}  p={pval:.4f}")

    return result


# ══════════════════════════════════════════════════════════════
# Task 4: 레짐 조건부 외생변수 시계열 생성
# ══════════════════════════════════════════════════════════════

def generate_ou_series(
    ou_regime : Dict[str, OUParams],
    rho       : float,
    cols      : Tuple[str, str],
    T         : int,
    dt        : float = 1 / 252,
    x0        : Optional[np.ndarray] = None,
    init_mode : str = "mean",
    seed      : Optional[int] = None,
) -> pd.DataFrame:
    """상관 충격을 가진 이변량 OU 시계열 생성.

    Σ = [[1, ρ], [ρ, 1]]  →  L = cholesky(Σ)  →  ε = L @ z, z ~ N(0,I)

    X1_t = X1_{t-1} + θ₁·(μ₁ - X1_{t-1})·Δt + σ₁·√Δt·ε₁
    X2_t = X2_{t-1} + θ₂·(μ₂ - X2_{t-1})·Δt + σ₂·√Δt·ε₂

    Parameters
    ----------
    ou_regime : {col: OUParams} 해당 레짐의 OU 모수 딕셔너리.
    rho       : 두 변수 잔차 간 상관계수.
    cols      : (col1, col2) 생성할 변수 쌍.
    T         : 생성 기간.
    dt        : 시간 간격. Default=1/252.
    x0        : 초기값 array (2,). None 이면 init_mode 사용.
    init_mode : 'mean' → OU 장기 평균,
                'last' → ou_regime 에서 마지막 관측 잔차 기반 (외부 주입 필요 시 x0 직접 지정).
    seed      : 재현성 시드.

    Returns
    -------
    pd.DataFrame : shape=(T, 2), columns=cols
    """
    col1, col2 = cols
    p1 = ou_regime[col1]
    p2 = ou_regime[col2]

    if x0 is not None:
        X1, X2 = float(x0[0]), float(x0[1])
    elif init_mode == "mean":
        X1, X2 = p1.mu, p2.mu
    else:
        raise ValueError(f"[generate_ou_series] 알 수 없는 init_mode: {init_mode}. "
                         "'mean' 또는 x0 직접 지정.")

    rho_c = float(np.clip(rho, -0.9999, 0.9999))
    Sigma  = np.array([[1.0, rho_c], [rho_c, 1.0]])
    L      = np.linalg.cholesky(Sigma)   # 하삼각

    rng      = np.random.default_rng(seed)
    sqrt_dt  = np.sqrt(dt)
    path1    = np.empty(T + 1)
    path2    = np.empty(T + 1)
    path1[0] = X1
    path2[0] = X2

    for t in range(T):
        z   = rng.standard_normal(2)
        eps = L @ z

        path1[t + 1] = (path1[t]
                        + p1.theta * (p1.mu - path1[t]) * dt
                        + p1.sigma * sqrt_dt * eps[0])
        path2[t + 1] = (path2[t]
                        + p2.theta * (p2.mu - path2[t]) * dt
                        + p2.sigma * sqrt_dt * eps[1])

    return pd.DataFrame({col1: path1[1:], col2: path2[1:]})


# ══════════════════════════════════════════════════════════════
# Task 5: 시나리오별 몬테카를로 시뮬레이션
# ══════════════════════════════════════════════════════════════

def _simulate_single_path(
    gap_ou    : Dict[str, OUParams],
    kp_ou     : Dict[str, OUParams],
    gap_rho   : float,
    kp_rho    : float,
    gap_params: GAPSimParams,
    kp_params : KPSimParams,
    T         : int,
    dt        : float,
    x0_gap    : Optional[np.ndarray],
    x0_kp     : Optional[np.ndarray],
    init_mode : str,
    rng       : np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 몬테카를로 경로 생성.

    Returns
    -------
    (gap_path, kp_path, etf_path) : 각각 shape=(T,)
    """
    seed_gap = int(rng.integers(1_000_000))
    seed_kp  = int(rng.integers(1_000_000))

    # ── 외생변수 시계열 생성 (Task 4) ───────────────────────
    df_gap_exog = generate_ou_series(
        ou_regime=gap_ou, rho=gap_rho, cols=GAP_COLS,
        T=T, dt=dt, x0=x0_gap, init_mode=init_mode, seed=seed_gap,
    )
    df_kp_exog = generate_ou_series(
        ou_regime=kp_ou, rho=kp_rho, cols=KP_COLS,
        T=T, dt=dt, x0=x0_kp, init_mode=init_mode, seed=seed_kp,
    )

    si_arr  = df_gap_exog["global_btc_svi"].values
    vix_arr = df_gap_exog["Global_RV"].values
    vol_arr = df_kp_exog["btc_volume_btc"].values
    kos_arr = df_kp_exog["VKOSPI"].values

    # ── GAP 경로 생성 ────────────────────────────────────────
    sqrt_dt   = np.sqrt(dt)
    gap_path  = np.empty(T + 1)
    kp_path   = np.empty(T + 1)
    gap_path[0] = gap_params.g0
    kp_path[0]  = kp_params.kp0

    z_gap = rng.standard_normal(T)
    z_kp  = rng.standard_normal(T)

    for t in range(T):
        sig_gap = gap_params.sigma0 * np.exp(
            gap_params.delta1 * si_arr[t] + gap_params.delta2 * vix_arr[t]
        )
        gap_path[t + 1] = (gap_path[t]
                           + gap_params.kappa * (gap_params.mu - gap_path[t]) * dt
                           + sig_gap * sqrt_dt * z_gap[t])

        sig_kp = kp_params.sigma0 * np.exp(
            kp_params.delta1 * vol_arr[t] + kp_params.delta2 * kos_arr[t]
        )
        kp_path[t + 1] = (kp_path[t]
                          + kp_params.kappa * (kp_params.mu - kp_path[t]) * dt
                          + sig_kp * sqrt_dt * z_kp[t])

    # ETF_t = NAV_0 × (1 + GAP_t)  (NAV_0=1 정규화 기준)
    etf_path = 1.0 + gap_path[1:]

    return gap_path[1:], kp_path[1:], etf_path


def monte_carlo_simulation(
    gap_ou_params   : Dict[int, Dict[str, OUParams]],
    gap_correlations: Dict[int, Dict[str, float]],
    kp_ou_params    : Dict[int, Dict[str, OUParams]],
    kp_correlations : Dict[int, Dict[str, float]],
    gap_sim_params  : GAPSimParams,
    kp_sim_params   : KPSimParams,
    scenarios       : Dict[str, Dict[str, int]],
    N               : int = 1000,
    T               : int = 252,
    dt              : float = 1 / 252,
    x0_gap          : Optional[np.ndarray] = None,
    x0_kp           : Optional[np.ndarray] = None,
    init_mode       : str = "mean",
    seed            : Optional[int] = None,
    save_dir        : Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """시나리오별 몬테카를로 ETF 가격 경로 시뮬레이션.

    Parameters
    ----------
    gap_ou_params   : estimate_ou_params() — GAP 레짐별 OU 모수.
    gap_correlations: estimate_correlations() — GAP 레짐별 잔차 상관.
    kp_ou_params    : estimate_ou_params() — KP 레짐별 OU 모수.
    kp_correlations : estimate_correlations() — KP 레짐별 잔차 상관.
    gap_sim_params  : GAP 과정 시뮬레이션 파라미터 (κ, μ, σ₀, δ₁, δ₂, g₀).
    kp_sim_params   : KP 과정 시뮬레이션 파라미터.
    scenarios       : {scenario_name: {"gap": gap_regime, "kp": kp_regime}}.
                      예: {"scenario_1": {"gap": 0, "kp": 0},
                           "scenario_2": {"gap": 1, "kp": 1}}
    N               : 몬테카를로 반복 수. Default=1000.
    T               : 시뮬레이션 기간. Default=252.
    dt              : 시간 간격. Default=1/252.
    x0_gap          : GAP 외생변수 초기값 (2,). None 이면 init_mode 적용.
    x0_kp           : KP 외생변수 초기값 (2,). None 이면 init_mode 적용.
    init_mode       : 'mean' (OU 장기 평균) 또는 x0 직접 지정.
    seed            : 재현성 시드.
    save_dir        : 결과 저장 디렉토리. None 이면 저장 안 함.

    Returns
    -------
    dict : {scenario_name: {"gap": (N,T), "kp": (N,T), "etf": (N,T)}}
    """
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    rng     = np.random.default_rng(seed)
    results : Dict[str, Dict[str, np.ndarray]] = {}

    for scenario_name, regime_map in scenarios.items():
        gap_r = regime_map["gap"]
        kp_r  = regime_map["kp"]

        if gap_r not in gap_ou_params:
            raise KeyError(f"[MC] scenario '{scenario_name}': GAP regime {gap_r} 없음.")
        if kp_r not in kp_ou_params:
            raise KeyError(f"[MC] scenario '{scenario_name}': KP regime {kp_r} 없음.")

        gap_ou  = gap_ou_params[gap_r]
        kp_ou   = kp_ou_params[kp_r]
        gap_rho = gap_correlations.get(gap_r, {}).get("GAP", 0.0)
        kp_rho  = kp_correlations.get(kp_r,  {}).get("KP",  0.0)

        gap_mat = np.zeros((N, T))
        kp_mat  = np.zeros((N, T))
        etf_mat = np.zeros((N, T))

        print(f"\n[MC] '{scenario_name}'  gap_regime={gap_r}  kp_regime={kp_r}  "
              f"gap_ρ={gap_rho:.3f}  kp_ρ={kp_rho:.3f}")

        for n in range(N):
            g, k, e = _simulate_single_path(
                gap_ou=gap_ou, kp_ou=kp_ou,
                gap_rho=gap_rho, kp_rho=kp_rho,
                gap_params=gap_sim_params, kp_params=kp_sim_params,
                T=T, dt=dt,
                x0_gap=x0_gap, x0_kp=x0_kp,
                init_mode=init_mode, rng=rng,
            )
            gap_mat[n] = g
            kp_mat[n]  = k
            etf_mat[n] = e

            if (n + 1) % 200 == 0:
                print(f"  {n+1}/{N} 완료 ...", flush=True)

        results[scenario_name] = {"gap": gap_mat, "kp": kp_mat, "etf": etf_mat}

        if save_dir:
            sd = Path(save_dir) / scenario_name
            sd.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(etf_mat).to_csv(sd / "etf_paths.csv", index=False)
            pd.DataFrame(gap_mat).to_csv(sd / "gap_paths.csv", index=False)
            pd.DataFrame(kp_mat).to_csv(sd / "kp_paths.csv",  index=False)
            print(f"[MC] '{scenario_name}' 저장 → {sd}")

    return results


# ══════════════════════════════════════════════════════════════
# Task 6: 검증 및 결과 비교
# ══════════════════════════════════════════════════════════════

def _mdd(path: np.ndarray) -> float:
    """단일 경로의 최대 낙폭(MDD) 계산."""
    cummax = np.maximum.accumulate(path)
    dd     = (path - cummax) / np.where(cummax != 0, cummax, 1.0)
    return float(dd.min())


def _risk_metrics(mat: np.ndarray, var_levels: Tuple[float, ...] = (0.05, 0.01)) -> dict:
    """N×T 경로 행렬에서 말기(t=T) 분포 기반 위험 지표 계산.

    Returns
    -------
    dict : mean_final, std_final, VaR_{q}, CVaR_{q}, mean_MDD
    """
    finals = mat[:, -1]                     # 말기 값 (N,)
    metrics: dict = {
        "mean_final": float(finals.mean()),
        "std_final" : float(finals.std(ddof=1)),
        "mean_MDD"  : float(np.mean([_mdd(mat[n]) for n in range(len(mat))])),
    }
    for q in var_levels:
        var  = float(np.percentile(finals, q * 100))
        cvar = float(finals[finals <= var].mean()) if (finals <= var).any() else var
        metrics[f"VaR_{int(q*100)}pct"]  = var
        metrics[f"CVaR_{int(q*100)}pct"] = cvar
    return metrics


def validate_and_compare(
    ou_params_gap   : Dict[int, Dict[str, OUParams]],
    ou_params_kp    : Dict[int, Dict[str, OUParams]],
    scenario_results: Dict[str, Dict[str, np.ndarray]],
    scenarios       : Dict[str, Dict[str, int]],
    df_exog         : pd.DataFrame,
    gap_regime_labels: pd.Series,
    kp_regime_labels : pd.Series,
    lags            : int = 20,
    save_dir        : Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """모수 추정 검증 + 시나리오 간 비교.

    검증 항목:
    1. 모수 추정 검증: 생성 시계열의 평균·분산이 OU 모수와 일치하는지
    2. ACF 플롯: 생성 시계열 vs 원본 레짐 구간
    3. ADF 정상성 검정
    4. 시나리오 간 위험 지표 비교 (VaR, CVaR, MDD 등)
    5. ETF / GAP / KP 분포 비교 시각화

    Parameters
    ----------
    ou_params_gap    : GAP 그룹 OU 모수.
    ou_params_kp     : KP 그룹 OU 모수.
    scenario_results : monte_carlo_simulation() 출력.
    scenarios        : 시나리오 정의 딕셔너리.
    df_exog          : 원본 외생변수 DataFrame.
    gap_regime_labels: GAP HMM 레짐 레이블.
    kp_regime_labels : KP HMM 레짐 레이블.
    lags             : ACF 최대 lag. Default=20.
    save_dir         : 결과 저장 디렉토리.

    Returns
    -------
    (param_check_df, adf_df, risk_df)
    """
    val_dir   = Path(save_dir) / "validation" if save_dir else None
    scen_dir  = Path(save_dir) / "scenarios"  if save_dir else None
    plots_dir = scen_dir / "plots" if scen_dir else None
    for d in [val_dir, scen_dir, plots_dir]:
        if d:
            d.mkdir(parents=True, exist_ok=True)

    # ── 1. 모수 추정 검증 ────────────────────────────────────
    param_rows: List[dict] = []
    adf_rows  : List[dict] = []

    all_ou = {"GAP": (ou_params_gap, gap_regime_labels, list(GAP_COLS)),
              "KP" : (ou_params_kp,  kp_regime_labels,  list(KP_COLS))}

    for group, (ou_params, regime_labels, cols) in all_ou.items():
        common = df_exog.index.intersection(regime_labels.index)
        df_g   = df_exog.loc[common, cols]
        lab    = regime_labels.loc[common]

        for regime, col_params in ou_params.items():
            mask = (lab == regime)
            for col, params in col_params.items():
                orig_vals = df_g.loc[mask, col].dropna().values

                param_rows.append({
                    "group"    : group,
                    "regime"   : regime,
                    "variable" : col,
                    "ou_mu"    : params.mu,
                    "ou_theta" : params.theta,
                    "ou_sigma" : params.sigma,
                    "ou_r2"    : params.r2,
                    "orig_mean": float(orig_vals.mean()),
                    "orig_std" : float(orig_vals.std(ddof=1)),
                    "n_obs"    : params.n_obs,
                })

                # ADF on original regime series
                try:
                    adf_stat, adf_p, *_ = adfuller(orig_vals, autolag="AIC")
                except Exception:
                    adf_stat, adf_p = np.nan, np.nan

                adf_rows.append({
                    "group"   : group,
                    "regime"  : regime,
                    "variable": col,
                    "adf_stat": float(adf_stat),
                    "adf_p"   : float(adf_p),
                    "stationary": adf_p < 0.05 if not np.isnan(adf_p) else None,
                })

                # ACF 플롯
                if len(orig_vals) > lags + 2:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                    sm_plot_acf(orig_vals, lags=lags, ax=axes[0], alpha=0.05)
                    axes[0].set_title(f"[원본] {col} | regime={regime}", fontsize=8)

                    # OU 이론 ACF: corr(lag) = exp(-θ·lag·Δt)
                    th = max(params.theta, 1e-6)
                    lag_arr = np.arange(0, lags + 1)
                    theory_acf = np.exp(-th * lag_arr / 252)
                    axes[1].bar(lag_arr, theory_acf, color="#4d9de0", alpha=0.7,
                                label="OU 이론 ACF")
                    axes[1].set_ylim(-0.1, 1.1)
                    axes[1].set_title(f"[이론] OU ACF  θ={th:.3f}", fontsize=8)
                    axes[1].legend(fontsize=7)
                    fig.suptitle(
                        f"ACF 비교 | {group} | {col} | regime={regime}",
                        fontsize=9, fontweight="bold")
                    fig.tight_layout()

                    if val_dir:
                        fp = val_dir / f"acf_{group}_{col}_r{regime}.png"
                        fig.savefig(fp, dpi=130, bbox_inches="tight")
                    plt.show()

    param_df = pd.DataFrame(param_rows).set_index(["group", "regime", "variable"])
    adf_df   = pd.DataFrame(adf_rows).set_index(["group", "regime", "variable"])

    if val_dir:
        param_df.to_csv(val_dir / "parameter_check.csv")
        adf_df.to_csv(val_dir / "adf_results.csv")
        print(f"[Task6] 검증 결과 → {val_dir}")

    print("\n[Task6] 모수 추정 검증:")
    print(param_df.to_string())

    # ── 2. 시나리오 간 비교 ──────────────────────────────────
    risk_rows: List[dict] = []

    for scenario_name, mats in scenario_results.items():
        gap_regime = scenarios[scenario_name]["gap"]
        kp_regime  = scenarios[scenario_name]["kp"]

        for key, mat in mats.items():
            m = _risk_metrics(mat)
            m.update({
                "scenario" : scenario_name,
                "series"   : key,
                "gap_regime": gap_regime,
                "kp_regime" : kp_regime,
            })
            risk_rows.append(m)

        # 개별 시나리오 분포 저장
        if scen_dir:
            etf_finals = mats["etf"][:, -1]
            pd.DataFrame({
                "etf_final": etf_finals,
                "gap_final": mats["gap"][:, -1],
                "kp_final" : mats["kp"][:, -1],
            }).to_csv(scen_dir / f"{scenario_name}_etf_distribution.csv", index=False)

    risk_df = pd.DataFrame(risk_rows).set_index(["scenario", "series"])

    if scen_dir:
        risk_df.to_csv(scen_dir / "risk_metrics_comparison.csv")

    print("\n[Task6] 시나리오 위험 지표:")
    print(risk_df.to_string())

    # ── 3. 비교 시각화 ───────────────────────────────────────
    _plot_comparison(scenario_results, save_dir=str(plots_dir) if plots_dir else None)

    return param_df, adf_df, risk_df


def _plot_comparison(
    scenario_results: Dict[str, Dict[str, np.ndarray]],
    save_dir        : Optional[str] = None,
) -> None:
    """시나리오별 ETF / GAP / KP 분포 비교 시각화."""
    scenarios = list(scenario_results.keys())
    colors    = plt.cm.tab10(np.linspace(0, 0.7, len(scenarios)))

    # ── ETF 경로 비교 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, mats) in enumerate(scenario_results.items()):
        etf = mats["etf"]                   # (N, T)
        t   = np.arange(etf.shape[1])
        p50 = np.percentile(etf, 50, axis=0)
        p25 = np.percentile(etf, 25, axis=0)
        p75 = np.percentile(etf, 75, axis=0)
        p05 = np.percentile(etf, 5,  axis=0)
        p95 = np.percentile(etf, 95, axis=0)
        ax.plot(t, p50, color=colors[i], linewidth=1.8, label=f"{name} (중위)")
        ax.fill_between(t, p25, p75, color=colors[i], alpha=0.20, label=f"{name} 50%")
        ax.fill_between(t, p05, p95, color=colors[i], alpha=0.08, label=f"{name} 90%")
    ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("기간 (일)")
    ax.set_ylabel("ETF 가격 (정규화, NAV₀=1)")
    ax.set_title("시나리오별 ETF 가격 경로 비교", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "etf_paths_comparison.png", dpi=130, bbox_inches="tight")
    plt.show()

    # ── VaR / CVaR 막대 비교 ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x     = np.arange(len(scenarios))
    width = 0.35
    var5  = [_risk_metrics(scenario_results[s]["etf"])["VaR_5pct"]  for s in scenarios]
    cvar5 = [_risk_metrics(scenario_results[s]["etf"])["CVaR_5pct"] for s in scenarios]
    var1  = [_risk_metrics(scenario_results[s]["etf"])["VaR_1pct"]  for s in scenarios]
    cvar1 = [_risk_metrics(scenario_results[s]["etf"])["CVaR_1pct"] for s in scenarios]

    axes[0].bar(x - width / 2, var5,  width, label="VaR 5%",  alpha=0.75, color="#4e79a7")
    axes[0].bar(x + width / 2, cvar5, width, label="CVaR 5%", alpha=0.75, color="#e15759")
    axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios, rotation=20)
    axes[0].set_title("VaR / CVaR (5%) 비교", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(True, linestyle="--", alpha=0.3, axis="y")

    axes[1].bar(x - width / 2, var1,  width, label="VaR 1%",  alpha=0.75, color="#4e79a7")
    axes[1].bar(x + width / 2, cvar1, width, label="CVaR 1%", alpha=0.75, color="#e15759")
    axes[1].set_xticks(x); axes[1].set_xticklabels(scenarios, rotation=20)
    axes[1].set_title("VaR / CVaR (1%) 비교", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.suptitle("시나리오별 ETF 위험 지표 비교", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "var_comparison.png", dpi=130, bbox_inches="tight")
    plt.show()

    # ── GAP / KP 말기 분포 ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, mats) in enumerate(scenario_results.items()):
        axes[0].hist(mats["gap"][:, -1], bins=50, density=True, alpha=0.45,
                     color=colors[i], label=name)
        axes[1].hist(mats["kp"][:, -1],  bins=50, density=True, alpha=0.45,
                     color=colors[i], label=name)

    for ax, title in zip(axes, ["GAP 말기 분포", "KP 말기 분포"]):
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("값")
        ax.set_ylabel("밀도")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("시나리오별 GAP / KP 말기 분포 비교", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "gap_kp_distribution.png", dpi=130, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 편의 파이프라인
# ══════════════════════════════════════════════════════════════

def run_simulation_pipeline(
    df_exog          : pd.DataFrame,
    gap_regime_labels: pd.Series,
    kp_regime_labels : pd.Series,
    gap_sim_params   : GAPSimParams,
    kp_sim_params    : KPSimParams,
    scenarios        : Dict[str, Dict[str, int]],
    N                : int = 1000,
    T                : int = 252,
    dt               : float = 1 / 252,
    x0_gap           : Optional[np.ndarray] = None,
    x0_kp            : Optional[np.ndarray] = None,
    init_mode        : str = "mean",
    seed             : Optional[int] = None,
    save_dir         : Optional[str] = None,
    validate         : bool = True,
) -> dict:
    """레짐 조건부 시뮬레이션 파이프라인 전체 실행.

    Task 1 → Task 2 → Task 3 → Task 5 → (Task 6) 순차 실행.

    Parameters
    ----------
    df_exog          : 외생변수 DataFrame.
                       필수 컬럼: global_btc_svi, Global_RV, btc_volume_btc, VKOSPI.
    gap_regime_labels: Global_RV HMM Viterbi 상태 Series.
    kp_regime_labels : VKOSPI_resid HMM Viterbi 상태 Series.
    gap_sim_params   : GAP 과정 파라미터 (κ, μ, σ₀, δ₁, δ₂, g₀).
    kp_sim_params    : KP 과정 파라미터 (κ, μ, σ₀, δ₁, δ₂, kp₀).
    scenarios        : {name: {"gap": gap_regime_num, "kp": kp_regime_num}}.
                       예: {"scenario_1": {"gap": 0, "kp": 0}}
    N                : 몬테카를로 반복 수. Default=1000.
    T                : 시뮬레이션 기간(일). Default=252.
    dt               : 시간 간격. Default=1/252.
    x0_gap           : GAP 외생변수 초기값 (global_btc_svi, Global_RV). None=init_mode 사용.
    x0_kp            : KP 외생변수 초기값 (btc_volume_btc, VKOSPI). None=init_mode 사용.
    init_mode        : 'mean' → OU 장기 평균으로 초기화.
    seed             : 재현성 시드.
    save_dir         : 결과 저장 루트 디렉토리.
    validate         : Task 6 검증·비교 실행 여부. Default=True.

    Returns
    -------
    {
      "gap_pairs"         : Dict[int, Dict[str, List]],
      "kp_pairs"          : Dict[int, Dict[str, List]],
      "gap_ou_params"     : Dict[int, Dict[str, OUParams]],
      "kp_ou_params"      : Dict[int, Dict[str, OUParams]],
      "gap_correlations"  : Dict[int, Dict[str, float]],
      "kp_correlations"   : Dict[int, Dict[str, float]],
      "scenario_results"  : Dict[str, Dict[str, np.ndarray]],
      "validation"        : (param_df, adf_df, risk_df) or None,
    }
    """
    print(f"\n{'═'*60}")
    print(f"  Regime-Conditioned Parametric Simulation")
    print(f"{'═'*60}")
    print(f"  GAP cols  : {GAP_COLS}")
    print(f"  KP cols   : {KP_COLS}")
    print(f"  Scenarios : {list(scenarios.keys())}")
    print(f"  N={N}  T={T}  dt={dt:.6f}  seed={seed}")

    GROUPS: Dict[str, Tuple[str, str]] = {
        "GAP": GAP_COLS,
        "KP" : KP_COLS,
    }

    # Task 1 ─ 연속 쌍 추출
    print(f"\n[Task 1] 레짐별 내부 연속 쌍 추출 ...")
    gap_pairs = extract_consecutive_pairs(df_exog, gap_regime_labels, list(GAP_COLS))
    kp_pairs  = extract_consecutive_pairs(df_exog, kp_regime_labels,  list(KP_COLS))

    # Task 2 ─ OU 모수 추정
    print(f"\n[Task 2] OU 모수 추정 ...")
    gap_ou_params = estimate_ou_params(gap_pairs, dt=dt)
    kp_ou_params  = estimate_ou_params(kp_pairs,  dt=dt)

    # Task 3 ─ 상관관계 추정
    print(f"\n[Task 3] 잔차 상관관계 추정 ...")
    gap_correlations = estimate_correlations(gap_ou_params, {"GAP": GAP_COLS})
    kp_correlations  = estimate_correlations(kp_ou_params,  {"KP":  KP_COLS})

    # Task 5 ─ 몬테카를로 시뮬레이션
    print(f"\n[Task 5] 몬테카를로 시뮬레이션 ...")
    mc_save = str(Path(save_dir) / "scenarios") if save_dir else None
    scenario_results = monte_carlo_simulation(
        gap_ou_params    = gap_ou_params,
        gap_correlations = gap_correlations,
        kp_ou_params     = kp_ou_params,
        kp_correlations  = kp_correlations,
        gap_sim_params   = gap_sim_params,
        kp_sim_params    = kp_sim_params,
        scenarios        = scenarios,
        N=N, T=T, dt=dt,
        x0_gap=x0_gap, x0_kp=x0_kp,
        init_mode=init_mode, seed=seed,
        save_dir=mc_save,
    )

    # Task 6 ─ 검증 및 비교
    val = None
    if validate:
        print(f"\n[Task 6] 검증 및 시나리오 비교 ...")
        val = validate_and_compare(
            ou_params_gap    = gap_ou_params,
            ou_params_kp     = kp_ou_params,
            scenario_results = scenario_results,
            scenarios        = scenarios,
            df_exog          = df_exog,
            gap_regime_labels= gap_regime_labels,
            kp_regime_labels = kp_regime_labels,
            lags             = 20,
            save_dir         = save_dir,
        )

    print(f"\n{'═'*60}")
    print(f"  시뮬레이션 파이프라인 완료")
    print(f"{'═'*60}")

    return {
        "gap_pairs"        : gap_pairs,
        "kp_pairs"         : kp_pairs,
        "gap_ou_params"    : gap_ou_params,
        "kp_ou_params"     : kp_ou_params,
        "gap_correlations" : gap_correlations,
        "kp_correlations"  : kp_correlations,
        "scenario_results" : scenario_results,
        "validation"       : val,
    }


# ══════════════════════════════════════════════════════════════
# 독립 실행 예시
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from preprocessing.run_pipeline import load_data, step_har, step_hmm_vol

    # ── 데이터 로드 및 HMM 레짐 레이블 추출 ──────────────────
    df_daily, _ = load_data()
    har_result  = step_har(df_daily, save=False)
    hmm_results = step_hmm_vol(df_daily, har_result, n_init=5, B=200, save=False)

    # Global_RV HMM → GAP 레짐 레이블
    comp_rv   = hmm_results["Global_RV"]
    result_rv = {1: comp_rv.result_k1, 2: comp_rv.result_k2,
                 3: comp_rv.result_k3}[comp_rv.optimal_K]

    # VKOSPI_resid HMM → KP 레짐 레이블
    comp_vk   = hmm_results["VKOSPI_resid"]
    result_vk = {1: comp_vk.result_k1, 2: comp_vk.result_k2,
                 3: comp_vk.result_k3}[comp_vk.optimal_K]

    # df_daily 인덱스와 정렬
    df_exog = df_daily[list(ALL_EXOG_COLS)].dropna()

    gap_regime_labels = pd.Series(
        result_rv.states[: len(df_exog)],
        index=df_exog.index[: len(result_rv.states)],
        name="gap_regime",
    )
    kp_regime_labels = pd.Series(
        result_vk.states[: len(df_exog)],
        index=df_exog.index[: len(result_vk.states)],
        name="kp_regime",
    )

    # ── 시뮬레이터 파라미터 (예시 값 — 실제 추정값으로 교체) ──
    gap_params = GAPSimParams(
        kappa=2.0, mu=0.005, sigma0=0.01,
        delta1=0.05, delta2=0.03, g0=0.0,
    )
    kp_params = KPSimParams(
        kappa=1.5, mu=0.02, sigma0=0.015,
        delta1=0.02, delta2=0.04, kp0=0.0,
    )

    # ── 시나리오 정의 (최적 K 레짐 번호 기준) ─────────────────
    K_gap = comp_rv.optimal_K
    K_kp  = comp_vk.optimal_K
    SCENARIOS: Dict[str, Dict[str, int]] = {
        f"scenario_{r+1}": {"gap": r, "kp": r}
        for r in range(min(K_gap, K_kp))
    }

    # ── 파이프라인 실행 ───────────────────────────────────────
    out = run_simulation_pipeline(
        df_exog           = df_exog,
        gap_regime_labels = gap_regime_labels,
        kp_regime_labels  = kp_regime_labels,
        gap_sim_params    = gap_params,
        kp_sim_params     = kp_params,
        scenarios         = SCENARIOS,
        N                 = 500,
        T                 = 252,
        dt                = 1 / 252,
        init_mode         = "mean",
        seed              = 42,
        save_dir          = str(_ROOT / "results"),
        validate          = True,
    )
