"""
Gaussian HMM — 변동성 계열 상태 구간 분할
==========================================
대상 변수:
  - Global_RV    : BTC 실현 변동성 (gap_train["btc_volatility"], 수준값)
  - VKOSPI_resid : HAR 표준화 잔차

구현:
  hmmlearn.hmm.GaussianHMM 사용 — C 확장 forward/backward 알고리즘
  (순수 Python EM 대비 100~1000배 빠름)

추정:
  - Gaussian emission, diagonal covariance (1차원 데이터)
  - n_init ≥ 10 다중 초기값 — local optimum 회피
  - 수렴 기준: log-likelihood 변화 < 1e-4

K 결정:
  1. BIC (K=2 vs K=3)
  2. Bootstrap LRT, B=1000 (Davies Problem 대응)
  3. Regime 점유율 ≥ 10%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm as hmmlearn_hmm
from joblib import Parallel, delayed


# ══════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════

@dataclass
class HMMResult:
    """단일 K에 대한 Gaussian HMM 적합 결과."""

    K          : int
    loglik     : float
    aic        : float
    bic        : float
    n_params   : int
    pi         : np.ndarray   # (K,) 초기 분포
    A          : np.ndarray   # (K, K) 전이 행렬
    mu         : np.ndarray   # (K,) regime 평균  (μ 오름차순 정렬)
    sigma      : np.ndarray   # (K,) regime 표준편차
    states     : np.ndarray   # (T,) Viterbi 최적 상태
    gamma      : np.ndarray   # (T, K) 사후 확률
    occupancy  : np.ndarray   # (K,) regime 점유율
    series_name: str = ""
    n_obs      : int = 0

    def print_summary(self) -> None:
        K = self.K
        print(f"\n{'═'*58}")
        print(f"  Gaussian HMM  |  Series: {self.series_name}  |  K={K}")
        print(f"{'═'*58}")
        print(f"  Log-Likelihood : {self.loglik:>14.4f}")
        print(f"  AIC            : {self.aic:>14.4f}")
        print(f"  BIC            : {self.bic:>14.4f}")
        print(f"  N params       : {self.n_params}")
        print(f"\n  Regime Parameters (sorted by mean):")
        print(f"  {'Regime':<8} {'μ':>10} {'σ':>10} {'Occupancy':>12}")
        print(f"  {'-'*44}")
        for k in range(K):
            ok = "OK" if self.occupancy[k] >= 0.10 else "LOW"
            print(f"  {k:<8} {self.mu[k]:>10.4f} {self.sigma[k]:>10.4f} "
                  f"{self.occupancy[k]:>11.1%}  [{ok}]")
        print(f"\n  Transition Matrix A (row=from, col=to):")
        print(f"  {'':8}" + "".join(f"  →R{k}" for k in range(K)))
        for j in range(K):
            print(f"  R{j}→   " + "".join(f"  {self.A[j, k]:.3f}" for k in range(K)))


@dataclass
class HMMComparison:
    """K=1 vs K=2 vs K=3 Gaussian HMM 비교 결과."""

    result_k1             : HMMResult
    result_k2             : HMMResult
    result_k3             : HMMResult
    lrt_stat_k1k2         : float
    bootstrap_pvalue_k1k2 : float
    bootstrap_dist_k1k2   : np.ndarray   # (B,)
    lrt_stat              : float          # K=2 vs K=3
    bootstrap_pvalue      : float
    bootstrap_dist        : np.ndarray   # (B,)
    optimal_K             : int
    selection_reason      : str
    series_name           : str = ""

    def print_summary(self) -> None:
        r1, r2, r3 = self.result_k1, self.result_k2, self.result_k3
        print(f"\n{'═'*58}")
        print(f"  HMM Model Comparison  |  Series: {self.series_name}")
        print(f"{'═'*58}")
        print(f"\n  {'Metric':<22} {'K=1':>10} {'K=2':>10} {'K=3':>10}")
        print(f"  {'-'*52}")
        print(f"  {'Log-Likelihood':<22} {r1.loglik:>10.4f} {r2.loglik:>10.4f} {r3.loglik:>10.4f}")
        print(f"  {'AIC':<22} {r1.aic:>10.4f} {r2.aic:>10.4f} {r3.aic:>10.4f}")
        print(f"  {'BIC':<22} {r1.bic:>10.4f} {r2.bic:>10.4f} {r3.bic:>10.4f}")
        print(f"  {'N params':<22} {r1.n_params:>10d} {r2.n_params:>10d} {r3.n_params:>10d}")
        print(f"\n  Bootstrap LRT (H0: K=1  vs  H1: K=2):")
        print(f"    Observed statistic : {self.lrt_stat_k1k2:>8.4f}")
        print(f"    p-value  (B=1000)  : {self.bootstrap_pvalue_k1k2:>8.4f}")
        print(f"\n  Bootstrap LRT (H0: K=2  vs  H1: K=3):")
        print(f"    Observed statistic : {self.lrt_stat:>8.4f}")
        print(f"    p-value  (B=1000)  : {self.bootstrap_pvalue:>8.4f}")
        print(f"\n  Regime Occupancy (>= 10% required):")
        print(f"    K=1 Regime 0: {r1.occupancy[0]:.1%}  [OK]")
        for k in range(2):
            ok = "OK" if r2.occupancy[k] >= 0.10 else "FAIL"
            print(f"    K=2 Regime {k}: {r2.occupancy[k]:.1%}  [{ok}]")
        for k in range(3):
            ok = "OK" if r3.occupancy[k] >= 0.10 else "FAIL"
            print(f"    K=3 Regime {k}: {r3.occupancy[k]:.1%}  [{ok}]")
        print(f"\n  => Optimal K = {self.optimal_K}")
        print(f"     Reason   : {self.selection_reason}")

    def to_comparison_table(self) -> pd.DataFrame:
        r1, r2, r3 = self.result_k1, self.result_k2, self.result_k3
        return pd.DataFrame({
            "K=1": [r1.loglik, r1.aic, r1.bic, r1.n_params],
            "K=2": [r2.loglik, r2.aic, r2.bic, r2.n_params],
            "K=3": [r3.loglik, r3.aic, r3.bic, r3.n_params],
        }, index=["Log-Likelihood", "AIC", "BIC", "N_params"])


# ══════════════════════════════════════════════
# 내부 유틸리티
# ══════════════════════════════════════════════

def _n_params(K: int) -> int:
    """자유 파라미터 수: K means + K sigmas + K*(K-1) transitions + (K-1) initial."""
    return K + K + K * (K - 1) + (K - 1)


def _sort_result(
    pi: np.ndarray, A: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    states: np.ndarray, gamma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """μ 오름차순으로 regime 인덱스를 재정렬."""
    idx = np.argsort(mu)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))
    return (pi[idx], A[np.ix_(idx, idx)], mu[idx], sigma[idx],
            inv[states], gamma[:, idx])


def _build_result(
    model:       hmmlearn_hmm.GaussianHMM,
    y:           np.ndarray,
    K:           int,
    series_name: str,
) -> HMMResult:
    """hmmlearn 모델 → HMMResult 변환."""
    obs     = y.reshape(-1, 1)
    loglik  = model.score(obs)
    states  = model.predict(obs)
    gamma   = model.predict_proba(obs)

    mu    = model.means_.flatten()
    sigma = np.sqrt(model.covars_.flatten())
    pi    = model.startprob_.copy()
    A     = model.transmat_.copy()
    T     = len(y)
    np_   = _n_params(K)

    pi, A, mu, sigma, states, gamma = _sort_result(pi, A, mu, sigma, states, gamma)
    occupancy = np.bincount(states, minlength=K) / T

    return HMMResult(
        K=K, loglik=loglik,
        aic=-2 * loglik + 2 * np_,
        bic=-2 * loglik + np_ * np.log(T),
        n_params=np_,
        pi=pi, A=A, mu=mu, sigma=sigma,
        states=states, gamma=gamma, occupancy=occupancy,
        series_name=series_name, n_obs=T,
    )


# ══════════════════════════════════════════════
# 단일 HMM 적합
# ══════════════════════════════════════════════

def fit_hmm(
    y:           np.ndarray,
    K:           int,
    n_init:      int   = 10,
    max_iter:    int   = 300,
    tol:         float = 1e-4,
    seed:        int   = 42,
    series_name: str   = "",
) -> HMMResult:
    """다중 초기값으로 Gaussian HMM을 적합한다 (hmmlearn 백엔드).

    Parameters
    ----------
    y          : (T,) 관측 시계열
    K          : 상태 수
    n_init     : 초기값 반복 횟수
    max_iter   : EM 최대 반복
    tol        : 수렴 기준
    seed       : 재현성 seed
    """
    obs         = y.reshape(-1, 1)
    best_loglik = -np.inf
    best_model  = None

    for i in range(n_init):
        model = hmmlearn_hmm.GaussianHMM(
            n_components    = K,
            covariance_type = "diag",
            n_iter          = max_iter,
            tol             = tol,
            random_state    = seed + i,
            init_params     = "stmc",
            params          = "stmc",
            verbose         = False,
        )
        try:
            model.fit(obs)
            ll = model.score(obs)
            if ll > best_loglik:
                best_loglik = ll
                best_model  = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"HMM K={K} 적합 실패: 모든 초기값에서 수렴 오류")

    return _build_result(best_model, y, K, series_name)


# ══════════════════════════════════════════════
# Bootstrap LRT (Davies Problem) — 병렬 처리
# ══════════════════════════════════════════════

def _boot_one(
    T:      int,
    pi:     np.ndarray,
    A:      np.ndarray,
    mu:     np.ndarray,
    var:    np.ndarray,
    n_init: int,
    seed:   int,
    k_null: int = 2,
    k_alt:  int = 3,
) -> float:
    """단일 bootstrap 반복 — joblib worker 함수 (top-level 필수).

    k_null 모델로 데이터를 생성한 뒤 k_null 과 k_alt 를 적합하여 LRT 통계량 반환.
    """
    import warnings
    warnings.filterwarnings("ignore")

    gen = hmmlearn_hmm.GaussianHMM(n_components=k_null, covariance_type="diag")
    gen.startprob_ = pi
    gen.transmat_  = A
    gen.means_     = mu.reshape(-1, 1)
    gen.covars_    = var.reshape(-1, 1)

    y_b, _ = gen.sample(T, random_state=seed)
    y_b    = y_b.flatten()

    ll_null = fit_hmm(y_b, K=k_null, n_init=n_init, seed=seed).loglik
    ll_alt  = fit_hmm(y_b, K=k_alt,  n_init=n_init, seed=seed + 1).loglik
    return -2.0 * (ll_null - ll_alt)


def bootstrap_lrt(
    y:           np.ndarray,
    result_null: HMMResult,
    result_alt:  HMMResult,
    B:           int   = 1000,
    n_init:      int   = 3,
    n_jobs:      int   = -1,
    seed:        int   = 0,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap LRT — H0: K=k_null  vs  H1: K=k_alt.  joblib 병렬 실행.

    Parameters
    ----------
    result_null : null 가설 모델 (K=k_null)
    result_alt  : 대립 가설 모델 (K=k_alt)
    n_jobs      : 병렬 worker 수 (-1 = 전체 코어)

    Returns
    -------
    lrt_stat, pvalue, bootstrap_dist : (B,)
    """
    T        = len(y)
    k_null   = result_null.K
    k_alt    = result_alt.K
    lrt_stat = -2.0 * (result_null.loglik - result_alt.loglik)
    rng      = np.random.default_rng(seed)
    seeds    = rng.integers(0, 10_000_000, size=B).tolist()

    boot_stats = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_boot_one)(
            T, result_null.pi, result_null.A,
            result_null.mu, result_null.sigma ** 2,
            n_init, int(s), k_null, k_alt,
        )
        for s in seeds
    )
    boot_stats = np.array(boot_stats)
    pvalue     = float(np.mean(boot_stats >= lrt_stat))
    return lrt_stat, pvalue, boot_stats


# ══════════════════════════════════════════════
# K=2 vs K=3 비교
# ══════════════════════════════════════════════

def compare_hmm(
    y:           np.ndarray,
    n_init:      int  = 10,
    B:           int  = 1000,
    seed:        int  = 42,
    series_name: str  = "",
) -> HMMComparison:
    """K=1·K=2·K=3 HMM 적합 후 BIC·Bootstrap LRT·점유율로 최적 K 결정.

    선택 절차 (순차)
    ----------------
    1. K=1 vs K=2 bootstrap LRT + BIC:
       둘 다 K=2 를 지지하지 않으면 → K=1 (구간 없음)
    2. K=2 vs K=3 bootstrap LRT + BIC:
       둘 다 K=3 을 지지하고 점유율 조건 충족 → K=3, 아니면 → K=2
    """
    print(f"  Fitting K=1 ... (n_init={n_init})", flush=True)
    r1 = fit_hmm(y, K=1, n_init=n_init, seed=seed,     series_name=series_name)

    print(f"  Fitting K=2 ... (n_init={n_init})", flush=True)
    r2 = fit_hmm(y, K=2, n_init=n_init, seed=seed + 1, series_name=series_name)

    print(f"  Fitting K=3 ... (n_init={n_init})", flush=True)
    r3 = fit_hmm(y, K=3, n_init=n_init, seed=seed + 2, series_name=series_name)

    print(f"  Bootstrap LRT K=1 vs K=2 ... (B={B}, parallel)", flush=True)
    lrt_k1k2, pval_k1k2, boot_k1k2 = bootstrap_lrt(
        y, r1, r2, B=B, n_init=3, n_jobs=-1, seed=seed + 3
    )

    print(f"  Bootstrap LRT K=2 vs K=3 ... (B={B}, parallel)", flush=True)
    lrt_k2k3, pval_k2k3, boot_k2k3 = bootstrap_lrt(
        y, r2, r3, B=B, n_init=3, n_jobs=-1, seed=seed + 4
    )

    # ── 최적 K 결정 ─────────────────────────────
    # Step 1: K=1 vs K=2
    occupancy_ok_k2 = all(r2.occupancy >= 0.10)
    bic_k2_over_k1  = r2.bic < r1.bic
    lrt_rejects_k1  = pval_k1k2 < 0.05

    if not occupancy_ok_k2 or not bic_k2_over_k1 or not lrt_rejects_k1:
        optimal_K = 1
        if not occupancy_ok_k2:
            reason = "K=2 regime occupancy < 10%: single regime (K=1) selected"
        elif not bic_k2_over_k1:
            reason = f"BIC favors K=1 over K=2: single regime selected"
        else:
            reason = (f"BIC favors K=2 but Bootstrap LRT fails to reject K=1 "
                      f"(p={pval_k1k2:.3f}): K=1 selected (parsimony)")
    else:
        # Step 2: K=2 vs K=3
        occupancy_ok_k3 = all(r3.occupancy >= 0.10)
        bic_k3_over_k2  = r3.bic < r2.bic
        lrt_rejects_k2  = pval_k2k3 < 0.05

        if not occupancy_ok_k3:
            optimal_K = 2
            reason    = "K=3 regime occupancy < 10%: K=2 selected"
        elif bic_k3_over_k2 and lrt_rejects_k2:
            optimal_K = 3
            reason    = (f"BIC favors K=3 and Bootstrap LRT rejects K=2 "
                         f"(p={pval_k2k3:.3f})")
        elif bic_k3_over_k2 and not lrt_rejects_k2:
            optimal_K = 2
            reason    = (f"BIC favors K=3 but Bootstrap LRT fails to reject K=2 "
                         f"(p={pval_k2k3:.3f}): K=2 selected (parsimony)")
        elif not bic_k3_over_k2 and lrt_rejects_k2:
            optimal_K = 2
            reason    = (f"BIC favors K=2; LRT rejects K=2 (p={pval_k2k3:.3f}) "
                         f"but BIC penalty dominates: K=2 selected")
        else:
            optimal_K = 2
            reason    = f"BIC and Bootstrap LRT both favor K=2 (p={pval_k2k3:.3f})"

    return HMMComparison(
        result_k1=r1, result_k2=r2, result_k3=r3,
        lrt_stat_k1k2=lrt_k1k2, bootstrap_pvalue_k1k2=pval_k1k2,
        bootstrap_dist_k1k2=boot_k1k2,
        lrt_stat=lrt_k2k3, bootstrap_pvalue=pval_k2k3, bootstrap_dist=boot_k2k3,
        optimal_K=optimal_K, selection_reason=reason,
        series_name=series_name,
    )


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════

_REGIME_COLORS = ["#2c7bb6", "#d7191c", "#1a9641"]


def plot_hmm_result(
    result:    HMMResult,
    series:    pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """단일 K에 대한 HMM 적합 결과 시각화 (5 패널)."""
    K      = result.K
    T      = result.n_obs
    dates  = series.index[:T]
    y      = series.values[:T]
    colors = _REGIME_COLORS[:K]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_ts    = fig.add_subplot(gs[0, :])
    ax_dist  = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_trans = fig.add_subplot(gs[2, 0])
    ax_occ   = fig.add_subplot(gs[2, 1])

    # Panel 1: 시계열 + Viterbi regime 음영
    ax_ts.plot(dates, y, color="#333333", linewidth=0.8, alpha=0.7, zorder=2)
    for k in range(K):
        mask = result.states == k
        ax_ts.fill_between(dates, y.min(), y.max(), where=mask, alpha=0.25,
                            color=colors[k],
                            label=f"R{k}  μ={result.mu[k]:.3f}  σ={result.sigma[k]:.3f}")
    ax_ts.set_title(
        f"{result.series_name}  |  Gaussian HMM K={K}  |  Viterbi Regime Assignment",
        fontsize=10)
    ax_ts.set_ylabel("Value")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Regime 분포
    y_grid = np.linspace(y.min() - 0.5 * y.std(), y.max() + 0.5 * y.std(), 300)
    for k in range(K):
        mask = result.states == k
        ax_dist.hist(y[mask], bins=25, density=True, alpha=0.35,
                     color=colors[k], label=f"R{k} (n={mask.sum()})")
        pdf = (np.exp(-0.5 * ((y_grid - result.mu[k]) / result.sigma[k]) ** 2)
               / (result.sigma[k] * np.sqrt(2 * np.pi)))
        ax_dist.plot(y_grid, pdf, color=colors[k], linewidth=1.8)
    ax_dist.set_title("Regime Emission Distributions", fontsize=9)
    ax_dist.set_xlabel("Value")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: 사후 확률
    for k in range(K):
        ax_gamma.plot(dates, result.gamma[:T, k], color=colors[k],
                      linewidth=0.9, label=f"P(R{k}|y)", alpha=0.85)
    ax_gamma.set_title("Posterior Probabilities  γ_t(k)", fontsize=9)
    ax_gamma.set_ylabel("Probability")
    ax_gamma.set_ylim(-0.05, 1.05)
    ax_gamma.legend(fontsize=8, loc="upper right")
    ax_gamma.grid(True, linestyle="--", alpha=0.4)

    # Panel 4: 전이확률 행렬
    im = ax_trans.imshow(result.A, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax_trans.set_xticks(range(K))
    ax_trans.set_yticks(range(K))
    ax_trans.set_xticklabels([f"→R{k}" for k in range(K)])
    ax_trans.set_yticklabels([f"R{j}→" for j in range(K)])
    ax_trans.set_title("Transition Probability Matrix  A", fontsize=9)
    for j in range(K):
        for k in range(K):
            ax_trans.text(k, j, f"{result.A[j, k]:.3f}", ha="center", va="center",
                          fontsize=9, color="white" if result.A[j, k] > 0.6 else "black")
    plt.colorbar(im, ax=ax_trans, fraction=0.046)

    # Panel 5: Regime 점유율
    bars = ax_occ.bar(range(K), result.occupancy * 100, color=colors, alpha=0.75)
    ax_occ.axhline(y=10, color="red", linewidth=1.2, linestyle="--",
                   label="Min threshold (10%)")
    ax_occ.set_xticks(range(K))
    ax_occ.set_xticklabels([f"Regime {k}" for k in range(K)])
    ax_occ.set_ylabel("Occupancy (%)")
    ax_occ.set_title("Regime Occupancy", fontsize=9)
    ax_occ.legend(fontsize=8)
    ax_occ.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, occ in zip(bars, result.occupancy):
        ax_occ.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{occ:.1%}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Gaussian HMM  |  {result.series_name}  |  K={K}  |"
        f"  LogLik={result.loglik:.2f}  BIC={result.bic:.2f}",
        fontsize=11, fontweight="bold",
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[HMM] Plot saved -> {save_path}")
    return fig


def plot_hmm_comparison(
    comp:      HMMComparison,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """K=1 vs K=2 vs K=3 비교 시각화 (1×3 레이아웃).

    Panel 1 : Bootstrap LRT  H0=K=1 vs H1=K=2
    Panel 2 : Bootstrap LRT  H0=K=2 vs H1=K=3
    Panel 3 : BIC / AIC 막대 (K=1, K=2, K=3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    def _lrt_panel(ax, boot_dist, lrt_stat, pval, h0_label, h1_label):
        ax.hist(boot_dist, bins=50, color="#4d9de0", alpha=0.7, density=True,
                label=f"Bootstrap dist (H0: {h0_label})")
        ax.axvline(x=lrt_stat, color="#e15759", linewidth=2.0,
                   label=f"Observed = {lrt_stat:.3f}")
        p95 = np.percentile(boot_dist, 95)
        ax.axvline(x=p95, color="#f28e2b", linewidth=1.5, linestyle="--",
                   label=f"95th pct = {p95:.3f}")
        reject = pval < 0.05
        ax.set_title(
            f"Bootstrap LRT  H0:{h0_label} vs H1:{h1_label}\n"
            f"{comp.series_name}  p={pval:.3f}  "
            f"({'reject ' + h0_label if reject else 'fail to reject ' + h0_label})",
            fontsize=9)
        ax.set_xlabel("LRT Statistic")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    _lrt_panel(axes[0], comp.bootstrap_dist_k1k2, comp.lrt_stat_k1k2,
               comp.bootstrap_pvalue_k1k2, "K=1", "K=2")
    _lrt_panel(axes[1], comp.bootstrap_dist,      comp.lrt_stat,
               comp.bootstrap_pvalue,      "K=2", "K=3")

    ax3   = axes[2]
    x     = np.arange(3)
    width = 0.35
    bics  = [comp.result_k1.bic, comp.result_k2.bic, comp.result_k3.bic]
    aics  = [comp.result_k1.aic, comp.result_k2.aic, comp.result_k3.aic]
    bars_bic = ax3.bar(x - width / 2, bics, width, label="BIC",
                       color="#4e79a7", alpha=0.8)
    bars_aic = ax3.bar(x + width / 2, aics, width, label="AIC",
                       color="#f28e2b", alpha=0.8)
    # 최적 K 에 별표
    opt_idx = comp.optimal_K - 1
    ax3.text(opt_idx - width / 2, bics[opt_idx],
             "★", ha="center", va="bottom", fontsize=12, color="#4e79a7")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["K=1", "K=2", "K=3"])
    ax3.set_ylabel("Information Criterion")
    ax3.set_title(
        f"BIC / AIC  |  {comp.series_name}\nOptimal K = {comp.optimal_K}",
        fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(
        f"HMM Model Selection  |  {comp.series_name}  |  Optimal K={comp.optimal_K}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[HMM] Comparison plot saved -> {save_path}")
    return fig


# ══════════════════════════════════════════════
# 편의 파이프라인
# ══════════════════════════════════════════════

def run_hmm_pipeline(
    series:    pd.Series,
    n_init:    int  = 10,
    B:         int  = 1000,
    seed:      int  = 42,
    plot:      bool = True,
    save_dir:  Optional[str] = None,
    transform: Optional[str] = None,   # None | 'log' | 'log1p'
) -> HMMComparison:
    """단일 시계열에 대해 K=2·K=3 HMM 비교 파이프라인 실행.

    Parameters
    ----------
    transform : None | 'log' | 'log1p'
        None    — 변환 없음 (표준화 잔차 등 이미 정상화된 변수)
        'log'   — log(y)   : 양수 보장 변수 (Global_RV 등)
        'log1p' — log(1+y) : 0 포함 가능 양수 변수 (SVI, 거래량 등)
    """
    from pathlib import Path

    name    = series.name or "series"
    s_clean = series.dropna()
    y       = s_clean.values.astype(float)

    if transform == "log":
        if np.any(y <= 0):
            raise ValueError(f"[HMM] log transform: '{name}' has non-positive values")
        y      = np.log(y)
        name   = f"log({name})"
        s_plot = pd.Series(y, index=s_clean.index, name=name)
    elif transform == "log1p":
        if np.any(y < 0):
            raise ValueError(f"[HMM] log1p transform: '{name}' has negative values")
        y      = np.log1p(y)
        name   = f"log1p({name})"
        s_plot = pd.Series(y, index=s_clean.index, name=name)
    else:
        s_plot = s_clean

    print(f"\n{'═'*58}")
    print(f"  Gaussian HMM Pipeline  |  {name}  (T={len(y)})")
    print(f"{'═'*58}")

    comp = compare_hmm(y, n_init=n_init, B=B, seed=seed, series_name=name)
    comp.print_summary()

    if plot:
        safe = name.replace("(", "_").replace(")", "").replace(" ", "_")
        save_k1 = str(Path(save_dir) / f"hmm_{safe}_K1.png")         if save_dir else None
        save_k2 = str(Path(save_dir) / f"hmm_{safe}_K2.png")         if save_dir else None
        save_k3 = str(Path(save_dir) / f"hmm_{safe}_K3.png")         if save_dir else None
        save_cp = str(Path(save_dir) / f"hmm_{safe}_comparison.png") if save_dir else None
        plot_hmm_result(comp.result_k1, s_plot, save_path=save_k1)
        plot_hmm_result(comp.result_k2, s_plot, save_path=save_k2)
        plot_hmm_result(comp.result_k3, s_plot, save_path=save_k3)
        plot_hmm_comparison(comp, save_path=save_cp)
        plt.show()

    return comp


# ══════════════════════════════════════════════
# 독립 실행
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from preprocessing.har_vkospi import run_har_pipeline

    _DATA  = _ROOT / "dataset" / "train_ver1"
    kp_df  = pd.read_csv(_DATA / "kp_train.csv",  parse_dates=["Date"]).set_index("Date").sort_index()
    gap_df = pd.read_csv(_DATA / "gap_train.csv", parse_dates=["Date"]).set_index("Date").sort_index()

    global_rv      = gap_df["btc_volatility"].dropna()
    global_rv.name = "Global_RV"

    vkospi      = kp_df["KOSPI_Volatility"].dropna()
    vkospi.name = "VKOSPI"
    har_result  = run_har_pipeline(vkospi, plot=False)
    resid_z     = pd.Series(
        har_result.residuals_z,
        index=vkospi.index[22:22 + len(har_result.residuals_z)],
        name="VKOSPI_resid",
    )

    run_hmm_pipeline(global_rv, n_init=10, B=1000, plot=True, save_dir=None, transform="log")
    run_hmm_pipeline(resid_z,   n_init=10, B=1000, plot=True, save_dir=None, transform=None)
