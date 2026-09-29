"""
시뮬레이터 검증 지표 모듈

연구 프레임워크 Step 4(개별 시뮬레이터 검증) / Step 6(통합 시뮬레이터 검증)에서 쓰인다.
검증 지표는 프레임워크가 규정한 세 검정뿐이다 -> calculate_statistical_tests()
  - PIT-KS      : 시점 조건부 분포 적합
  - VaR-Kupiec  : VaR 초과 빈도
  - ES          : 꼬리 손실 크기 (모수적 부트스트랩으로 p-value 산출)

예측구간 커버리지(PICP/NMPIW/CovErr)는 제거했다. 예측구간 품질 평가 지표
(Khosravi et al. 2011)로 본 연구의 경로 생성 목적과 맞지 않고, ES 가 검정 능력을
갖게 되면서 꼬리 진단 역할도 ES 로 흡수됐다.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, chi2
from statsmodels.tsa.stattools import acf


def reference_distribution(simulated_paths, statistic_fn, n_ref=2000, seed=42):
    """
    모형 내재 기준분포 (모수적 부트스트랩 / 사후예측검정)

    "모형이 옳다"는 전제에서 어떤 통계량이 가질 수 있는 값들의 분포를 만든다.
    시뮬레이션 경로 하나를 관측치인 척 꺼내 통계량을 계산하는 일을 n_ref번 반복한다.

    플러그인 방식(모수를 추정값에 고정)이므로 추정 불확실성이 귀무분포에 반영되지
    않는다. 이는 검정을 보수적으로 — 기각을 덜 하게 — 만드는 알려진 성질이다.
    따라서 "통과"만 보지 말고 반환된 분포의 폭(percentile 5/50/95)을 함께 봐야 한다.
    분포가 지나치게 넓으면 그 검정은 사실상 검정력이 없다는 뜻이다.

    Args:
        simulated_paths: 몬테카를로 경로 (N x T)
        statistic_fn: (held_out_paths (M x T)) -> (M,) 통계량 배열.
                      경로별 통계량을 한 번에 계산하는 벡터화 함수여야 한다.
                      모형 쪽 요약량(밴드, VaR_t, ES_t 등)은 호출부에서 전체 경로로
                      한 번만 계산해 클로저로 넘긴다 — 매 반복 재계산하면
                      O(n_ref * N log N)이 되어 비용이 폭증한다.
        n_ref: 기준분포 표본 수
        seed: 표본 추출 시드 (재현성)

    Returns:
        np.ndarray: 유한한 통계량 값들 (길이 <= n_ref)
    """
    paths = np.asarray(simulated_paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] < 2:
        return np.array([])

    N = paths.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_ref, N), replace=False)

    try:
        vals = np.asarray(statistic_fn(paths[idx]), dtype=float).ravel()
    except Exception:
        return np.array([])
    return vals[np.isfinite(vals)]


def reference_percentile(observed, reference):
    """관측 통계량이 기준분포의 몇 백분위인지 (0~100). 비교 불가면 nan."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or observed is None or not np.isfinite(observed):
        return float("nan")
    return float(100.0 * np.mean(ref < observed))


def two_sided_pvalue(observed, reference):
    """
    기준분포 기반 양측 경험적 p-value.

    p = 2 * min(P(ref <= obs), P(ref >= obs)), 1로 절단.
    <= 와 >= 를 함께 쓰므로 동점(tie)이 있어도 p가 0으로 붕괴하지 않는다.
    tail_error 처럼 부호를 갖는 통계량은 반드시 양측으로 판정해야 한다.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or observed is None or not np.isfinite(observed):
        return float("nan")
    p_lo = float(np.mean(ref <= observed))
    p_hi = float(np.mean(ref >= observed))
    return float(min(1.0, 2.0 * min(p_lo, p_hi)))


def realized_volatility(returns, window=20):
    """
    Realized Volatility 계산 (rolling window)
    
    Args:
        returns: 수익률 시계열
        window: 윈도우 크기
    
    Returns:
        np.array: Realized Volatility 시계열
    """
    returns_series = pd.Series(returns)
    rv = returns_series.rolling(window=window).std() * np.sqrt(252)  # 연간화
    return rv.bfill().fillna(rv.iloc[window-1] if len(rv) > window-1 else rv.iloc[-1] if len(rv) > 0 else 0).values


def pit_ks_test(actual_returns, simulated_returns_paths):
    """
    PIT-KS 검정
    각 시점 t에서 u_t = ECDF_t(r_t)를 계산하고, u_t들이 U(0,1)인지 KS 검정
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 시뮬레이션 수익률 경로들 (N x T 배열)
    
    Returns:
        dict: {'ks_statistic': KS 통계량, 'ks_pvalue': p-value}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    if simulated_returns_paths.ndim == 1:
        simulated_returns_paths = simulated_returns_paths.reshape(1, -1)
    
    T = len(actual_returns)
    N = len(simulated_returns_paths)
    
    # 각 시점 t에서 u_t = ECDF_t(r_t) 계산
    u_t = np.zeros(T)
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_returns_paths[:, t]
        # 경험적 CDF: ECDF_t(r_t) = (# of sim_values_t <= r_t) / N
        u_t[t] = np.mean(sim_values_t <= actual_returns[t])
    
    # u_t들이 U(0,1)인지 KS 검정
    from scipy.stats import kstest
    ks_stat, ks_pvalue = kstest(u_t, 'uniform')
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue)
    }


def pit_ljungbox_test(actual_returns, simulated_returns_paths, lags=None):
    """
    PIT 시계열에 대한 Ljung–Box 검정 (ACF 검정)
    PIT u_t = ECDF_t(r_t)가 iid U(0,1)이면 자기상관이 0이어야 함.
    H0: PIT 시계열의 첫 h개 자기상관이 0 (잔차가 백색잡음)
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 시뮬레이션 수익률 경로들 (N x T 배열)
        lags: 검정에 사용할 lag 수 (None이면 min(10, T//2-1) 또는 [1,...,10])
    
    Returns:
        dict: {'lb_statistic': LB 통계량, 'lb_pvalue': p-value, 'lags': 사용된 lag}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    if simulated_returns_paths.ndim == 1:
        simulated_returns_paths = simulated_returns_paths.reshape(1, -1)
    
    T = len(actual_returns)
    N = len(simulated_returns_paths)
    
    # PIT u_t 계산 (pit_ks_test와 동일)
    u_t = np.zeros(T)
    for t in range(T):
        sim_values_t = simulated_returns_paths[:, t]
        u_t[t] = np.mean(sim_values_t <= actual_returns[t])
    
    # 경계 처리: u_t가 0 또는 1이면 변환 (LB는 연속형 가정)
    u_t = np.clip(u_t, 1e-6, 1 - 1e-6)
    
    if lags is None:
        lags = min(10, max(1, T // 2 - 1))
    h = int(lags) if np.isscalar(lags) else int(max(lags))
    h = max(1, min(h, T // 2 - 1))
    
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(u_t, lags=h, return_df=True)
        # 마지막 lag까지의 joint 검정 결과 사용
        lb_stat = float(lb['lb_stat'].iloc[-1])
        lb_pvalue = float(lb['lb_pvalue'].iloc[-1])
    except Exception:
        # 수동 계산: LB Q = n(n+2) * sum_{k=1}^{h} r_k^2 / (n-k), Q ~ chi2(h)
        acf_vals = acf(u_t - np.mean(u_t), nlags=h, fft=True)[1: h + 1]
        n = len(u_t)
        q = n * (n + 2) * np.sum(acf_vals**2 / (n - np.arange(1, h + 1)))
        lb_stat = float(q)
        lb_pvalue = float(1 - chi2.cdf(q, df=h))
    
    return {
        'lb_statistic': lb_stat,
        'lb_pvalue': lb_pvalue,
        'lags': h
    }


def var_kupiec_test(actual_returns, simulated_returns_paths, alpha=0.05):
    """
    VaR-Kupiec 검정
    각 시점 t에서 VaR_t = quantile_i(r_hat^{(i)}_t, alpha)를 계산하고,
    I_t = 1{r_t < VaR_t}로 Kupiec LR_uc 검정
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 시뮬레이션 수익률 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'lr_uc': LR 통계량, 'pvalue': p-value, 'exceedance_rate': 초과율}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    if simulated_returns_paths.ndim == 1:
        simulated_returns_paths = simulated_returns_paths.reshape(1, -1)
    
    T = len(actual_returns)
    
    # 각 시점 t에서 VaR_t 계산
    var_t = np.zeros(T)
    I_t = np.zeros(T, dtype=bool)
    
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_returns_paths[:, t]
        # VaR_t = quantile_i(r_hat^{(i)}_t, alpha)
        var_t[t] = np.quantile(sim_values_t, alpha)
        # I_t = 1{r_t < VaR_t}
        I_t[t] = actual_returns[t] < var_t[t]
    
    # Kupiec LR_uc 검정
    x = int(np.sum(I_t))  # 초과 횟수
    rate = x / T if T > 0 else 0.0  # 초과율
    
    # LLR = -2 * (log L(H0) - log L(H1)), H0: 초과율=alpha, H1: MLE=x/T
    # x=0 또는 x=T여도 공식 적용 (logpmf(0,T,0)=0, logpmf(T,T,1)=0)
    if T == 0:
        lr_uc = 0.0
        pvalue = 1.0
    else:
        log_l_h0 = stats.binom.logpmf(x, T, alpha)
        rate_mle = rate
        if rate_mle <= 0:
            log_l_h1 = 0.0
        elif rate_mle >= 1:
            log_l_h1 = 0.0
        else:
            log_l_h1 = stats.binom.logpmf(x, T, rate_mle)
        lr_uc = float(-2 * (log_l_h0 - log_l_h1))
        lr_uc = max(0.0, lr_uc)
        pvalue = float(1 - chi2.cdf(lr_uc, df=1))
    
    return {
        'lr_uc': float(lr_uc),
        'pvalue': float(pvalue),
        'exceedance_rate': float(rate),
        'expected_rate': alpha,
        'n_exceedances': int(x),
        'n_total': T
    }


def _es_var_curves(simulated_returns_paths, alpha):
    """시점별 VaR_t 와 ES_t 를 전체 경로에서 한 번만 계산한다.

    기준분포를 만들 때 경로마다 이걸 다시 구하면 O(n_ref * N log N)이 된다.
    leave-one-out 효과는 1/N 수준이라 무시할 수 있으므로 전체 경로로 한 번만 구해
    재사용한다 (reference_distribution 의 설계 전제와 같다).
    """
    paths = np.asarray(simulated_returns_paths, dtype=float)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    T = paths.shape[1]

    var_t = np.nanquantile(paths, alpha, axis=0)

    es_t = np.empty(T)
    for t in range(T):
        col = paths[:, t]
        col = col[np.isfinite(col)]
        if col.size == 0:
            es_t[t] = np.nan
            continue
        tail = col[col <= var_t[t]]
        # VaR 이하 표본이 0개면 최소 1개(최솟값) 강제 포함 — 기존 규약 유지
        es_t[t] = np.mean(tail) if tail.size else np.min(col)
    return var_t, es_t


def _tail_error_from(series, var_t, es_t):
    """단일 계열의 tail_error. 위반일이 없으면 nan."""
    r = np.asarray(series, dtype=float).ravel()
    n = min(len(r), len(var_t))
    r, v, e = r[:n], var_t[:n], es_t[:n]
    mask = np.isfinite(r) & np.isfinite(v) & np.isfinite(e) & (r < v)
    if not mask.any():
        return float("nan")
    return float(np.mean(r[mask]) - np.mean(e[mask]))


def _tail_error_batch(paths, var_t, es_t):
    """여러 경로의 tail_error 를 한 번에. (M x T) -> (M,)"""
    P = np.asarray(paths, dtype=float)
    n = min(P.shape[1], len(var_t))
    P, v, e = P[:, :n], var_t[:n], es_t[:n]
    mask = np.isfinite(P) & (P < v) & np.isfinite(v) & np.isfinite(e)
    cnt = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_r = np.where(cnt > 0, np.nansum(np.where(mask, P, 0.0), axis=1) / cnt, np.nan)
        mean_e = np.where(cnt > 0,
                          np.nansum(np.where(mask, np.broadcast_to(e, P.shape), 0.0), axis=1) / cnt,
                          np.nan)
    return mean_r - mean_e


def es_test(actual_returns, simulated_returns_paths, alpha=0.05,
            n_ref=2000, seed=42):
    """
    ES (Expected Shortfall) 검정 — 모수적 부트스트랩

    각 시점 t 에서
        VaR_t = quantile_i(r_hat^{(i)}_t, alpha)
        ES_t  = mean_i(r_hat^{(i)}_t | r_hat^{(i)}_t <= VaR_t)
    위반일 E = {t : r_t < VaR_t} 에 대해
        tail_error = mean(r_t | E) - mean(ES_t | E)

    부호 규약: 손실이 음수인 수익률 공간이므로
        tail_error > 0  실현 손실이 모형 예측보다 덜 심함 (꼬리 과대평가)
        tail_error < 0  실현 손실이 모형 예측보다 더 심함 (꼬리 과소평가)

    tail_error 자체는 귀무분포가 없어 그대로는 판정에 쓸 수 없다. 그래서 시뮬레이션
    경로 하나를 관측치인 척 놓고 같은 방식으로 tail_error 를 계산하는 일을 n_ref번
    반복해 모형 내재 기준분포를 만들고, 실제 계열의 값이 그 분포의 어디에 있는지로
    양측 검정한다. 부호를 갖는 통계량이므로 반드시 양측이다.

    참고 — Gneiting (2011)의 elicitability 결과 때문에 "ES는 백테스트할 수 없다"는
    오해가 있었으나 이는 모형 '선택'의 문제이지 '검정'의 문제가 아니다.
    Acerbi & Szekely (2014), Fissler, Ziegel & Gneiting (2016) 참조.
    Acerbi-Szekely 의 검정을 직접 구현하는 것도 대안으로 검토 가능하나, 여기서는
    이미 검증된 기준분포 경로를 재사용해 구현 위험을 줄였다.

    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 시뮬레이션 수익률 경로 (N x T)
        alpha: VaR 유의수준 (Kupiec 검정과 동일 값을 쓴다)
        n_ref: 기준분포 표본 수
        seed: 재현성 시드

    Returns:
        dict: tail_error, pvalue, percentile, ref_p5/p50/p95, n_violations 등
    """
    actual = np.asarray(actual_returns, dtype=float).ravel()
    paths = np.asarray(simulated_returns_paths, dtype=float)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)

    var_t, es_t = _es_var_curves(paths, alpha)

    n = min(len(actual), len(var_t))
    viol = np.isfinite(actual[:n]) & (actual[:n] < var_t[:n])
    n_viol = int(viol.sum())

    empty = {
        "tail_error": None, "mean_actual_es": None, "mean_sim_es": None,
        "n_violations": n_viol, "pvalue": float("nan"), "percentile": float("nan"),
        "ref_p5": float("nan"), "ref_p50": float("nan"), "ref_p95": float("nan"),
        "n_ref_effective": 0, "is_valid": False,
    }
    if n_viol == 0:
        # 위반일이 없으면 ES 는 정의되지 않는다 (0.0으로 두면 오해 소지)
        return empty

    mean_actual_es = float(np.mean(actual[:n][viol]))
    mean_sim_es = float(np.mean(es_t[:n][viol]))
    tail_error = mean_actual_es - mean_sim_es

    ref = reference_distribution(
        paths, lambda held: _tail_error_batch(held, var_t, es_t),
        n_ref=n_ref, seed=seed,
    )
    pval = two_sided_pvalue(tail_error, ref)
    pct = reference_percentile(tail_error, ref)

    return {
        "tail_error": float(tail_error),
        "mean_actual_es": mean_actual_es,
        "mean_sim_es": mean_sim_es,
        "n_violations": n_viol,
        "pvalue": pval,
        "percentile": pct,
        "ref_p5": float(np.percentile(ref, 5)) if ref.size else float("nan"),
        "ref_p50": float(np.percentile(ref, 50)) if ref.size else float("nan"),
        "ref_p95": float(np.percentile(ref, 95)) if ref.size else float("nan"),
        "n_ref_effective": int(ref.size),
        # 기준분포를 못 만들면(ref 비어 있음) 판정 불가 -> 통과로 처리하지 않는다
        "is_valid": bool(np.isfinite(pval) and pval >= 0.05),
    }


def calculate_statistical_tests(actual_returns, simulated_returns_paths, alpha=0.05):
    """
    통계적 검정 수행 (수익률 기준). PIT-KS, VaR-Kupiec, ES.
    NAV 역변환은 괴리율 결합 등 별도 용도로만 사용.
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 몬테카를로 시뮬레이션 수익률 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'pit_ks': {...}, 'kupiec': {...}, 'es': {...}, 'is_valid': bool}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    # 1. PIT-KS 검정
    pit_ks_result = pit_ks_test(actual_returns, simulated_returns_paths)
    
    # 2. VaR-Kupiec 검정
    kupiec_result = var_kupiec_test(actual_returns, simulated_returns_paths, alpha=alpha)
    
    # 3. ES 검정
    es_result = es_test(actual_returns, simulated_returns_paths, alpha=alpha)
    
    # 전체 유효성 판단 — 프레임워크가 규정한 세 검정 모두를 쓴다.
    # ES 는 이전에 p-value 가 없어 판정에서 빠져 있었으나, 모수적 부트스트랩으로
    # 귀무분포를 얻으면서 다른 두 검정과 동등하게 판정에 들어간다.
    is_valid = bool(
        pit_ks_result['ks_pvalue'] >= 0.05
        and kupiec_result['pvalue'] >= 0.05
        and es_result.get('is_valid', False)
    )
    
    return {
        'pit_ks': pit_ks_result,
        'kupiec': kupiec_result,
        'es': es_result,
        'is_valid': is_valid
    }
