"""
시뮬레이터 검증 지표 모듈

연구 프레임워크 Step 4(개별 시뮬레이터 검증) / Step 6(통합 시뮬레이터 검증)에서 쓰인다.
  - 본 검증: PIT-KS, VaR-Kupiec, ES  -> calculate_statistical_tests()
  - 보조 진단: 예측구간 커버리지(PICP/NMPIW) -> calculate_all_metrics()
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, chi2
from statsmodels.tsa.stattools import acf


def interval_coverage(actual, simulated_paths, levels=(0.50, 0.80, 0.95)):
    """
    예측구간 커버리지 (PICP, Prediction Interval Coverage Probability)

    각 시점 t에서 몬테카를로 경로의 분위수로 명목 (1-alpha) 예측구간을 구성하고,
    실제 관측치가 그 구간 안에 들어간 시점의 비율(경험적 커버리지)을 센다.
    경험적 커버리지가 명목수준에 가까울수록 분포 예측이 잘 보정된 것이다.

    커버리지만 보면 구간을 넓게 잡을수록 유리하므로, 구간폭을 실제 시계열의
    변동범위로 정규화한 NMPIW를 함께 보고한다(낮을수록 좁고 예리한 구간).

    Args:
        actual: 실제 시계열 (T,)
        simulated_paths: 몬테카를로 경로 (N x T)
        levels: 명목 신뢰수준 리스트

    Returns:
        dict:
            'picp'           {level: 경험적 커버리지}   명목수준에 가까울수록 좋음
            'coverage_error' mean |경험적 - 명목|        낮을수록 좋음
            'nmpiw'          {level: 정규화 평균 구간폭} 낮을수록 좋음
    """
    actual = np.asarray(actual, dtype=float)
    paths = np.asarray(simulated_paths, dtype=float)

    if paths.ndim != 2:
        raise ValueError(f"simulated_paths는 (N x T) 2차원이어야 한다: ndim={paths.ndim}")

    T = min(len(actual), paths.shape[1])
    actual = actual[:T]
    paths = paths[:, :T]

    valid = np.isfinite(actual)
    if valid.sum() == 0 or paths.shape[0] == 0:
        nan_by_level = {lv: np.nan for lv in levels}
        return {'picp': nan_by_level, 'coverage_error': np.nan, 'nmpiw': dict(nan_by_level)}

    # 구간폭 정규화 기준: 실제 시계열의 변동범위 (0이면 정규화 생략)
    span = np.nanmax(actual[valid]) - np.nanmin(actual[valid])

    picp, nmpiw, errors = {}, {}, []
    for lv in levels:
        alpha = 1.0 - lv
        lower = np.nanquantile(paths, alpha / 2.0, axis=0)
        upper = np.nanquantile(paths, 1.0 - alpha / 2.0, axis=0)

        inside = (actual >= lower) & (actual <= upper) & valid
        emp = inside.sum() / valid.sum()

        picp[lv] = float(emp)
        errors.append(abs(emp - lv))
        width = np.nanmean((upper - lower)[valid])
        nmpiw[lv] = float(width / span) if span > 0 else float('nan')

    return {
        'picp': picp,
        'coverage_error': float(np.mean(errors)),
        'nmpiw': nmpiw,
    }


def coverage_metrics(actual, simulated_paths, prefix, levels=(0.50, 0.80, 0.95)):
    """
    interval_coverage 결과를 평탄한 지표 dict로 변환한다.
    예) prefix='price' -> picp50_price, picp80_price, picp95_price,
                          coverage_error_price, nmpiw95_price
    """
    cov = interval_coverage(actual, simulated_paths, levels=levels)
    out = {f'coverage_error_{prefix}': cov['coverage_error']}
    for lv in levels:
        out[f'picp{int(round(lv * 100))}_{prefix}'] = cov['picp'][lv]
    out[f'nmpiw95_{prefix}'] = cov['nmpiw'].get(0.95, float('nan'))
    return out


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


def calculate_all_metrics(actual_nav, simulated_nav, actual_returns, simulated_returns,
                          monte_carlo_nav_paths=None, monte_carlo_returns_paths=None):
    """
    보조 검증 지표 계산 — 예측구간 커버리지

    본 검증(PIT-KS, VaR-Kupiec, ES)은 calculate_statistical_tests()가 담당한다.
    이 함수는 그 세 검정이 다루지 않는 축인 '구간 커버리지'만 보조로 산출한다.

    Args:
        actual_nav: 실제 가격/수준 시계열
        simulated_nav: 대표 경로 (커버리지 계산에는 쓰이지 않으며 서명 호환용)
        actual_returns: 실제 수익률
        simulated_returns: 대표 경로 수익률 (동일)
        monte_carlo_nav_paths: MC 가격 경로 (N x T) — 커버리지 산출에 필수
        monte_carlo_returns_paths: MC 수익률 경로 (N x T)

    Returns:
        dict: picp50/80/95_price, coverage_error_price, nmpiw95_price 및 동일한 _vol 계열
    """
    metrics = {}

    # 가격 예측구간 커버리지
    print("  [예측구간 커버리지] 계산 중...")
    if monte_carlo_nav_paths is not None and len(monte_carlo_nav_paths) > 0:
        if isinstance(monte_carlo_nav_paths, list):
            monte_carlo_nav_paths = np.array(monte_carlo_nav_paths)
        metrics.update(coverage_metrics(actual_nav, monte_carlo_nav_paths, 'price'))
    else:
        # 대표 경로 하나로는 구간을 만들 수 없다
        metrics.update(coverage_metrics(actual_nav, np.empty((0, len(actual_nav))), 'price'))

    # 변동성 예측구간 커버리지
    actual_rv = realized_volatility(actual_returns)
    if monte_carlo_returns_paths is not None and len(monte_carlo_returns_paths) > 0:
        if isinstance(monte_carlo_returns_paths, list):
            monte_carlo_returns_paths = np.array(monte_carlo_returns_paths)
        sim_vol_paths = np.array([realized_volatility(ret) for ret in monte_carlo_returns_paths])
        metrics.update(coverage_metrics(actual_rv, sim_vol_paths, 'vol'))
    else:
        metrics.update(coverage_metrics(actual_rv, np.empty((0, len(actual_rv))), 'vol'))

    return metrics


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


def es_test(actual_returns, simulated_returns_paths, alpha=0.05):
    """
    ES (Expected Shortfall) 검정
    각 시점 t에서 ES_t = mean_i(r_hat^{(i)}_t | r_hat^{(i)}_t <= VaR_t)를 계산하고,
    위반일 E={t: r_t < VaR_t}에서 tail_error = mean(r_t|E) - mean(ES_t|E) 계산
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 시뮬레이션 수익률 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'tail_error': tail_error, 'mean_actual_es': 실제 ES 평균, 'mean_sim_es': 시뮬레이션 ES 평균}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    if simulated_returns_paths.ndim == 1:
        simulated_returns_paths = simulated_returns_paths.reshape(1, -1)
    
    T = len(actual_returns)
    
    # 각 시점 t에서 VaR_t와 ES_t 계산
    var_t = np.zeros(T)
    es_t = np.zeros(T)
    I_t = np.zeros(T, dtype=bool)
    
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_returns_paths[:, t]
        # VaR_t = quantile_i(r_hat^{(i)}_t, alpha)
        var_t[t] = np.quantile(sim_values_t, alpha)
        # ES_t = mean_i(r_hat^{(i)}_t | r_hat^{(i)}_t <= VaR_t)
        tail_samples = sim_values_t[sim_values_t <= var_t[t]]
        
        # 예외처리: VaR 이하 표본이 0개면 최소 1개 강제 포함
        if len(tail_samples) == 0:
            # 가장 작은 값 1개 포함
            tail_samples = np.array([np.min(sim_values_t)])
        
        es_t[t] = np.mean(tail_samples)
        # I_t = 1{r_t < VaR_t}
        I_t[t] = actual_returns[t] < var_t[t]
    
    # 위반일 E = {t: r_t < VaR_t}
    E = np.where(I_t)[0]
    
    if len(E) == 0:
        # 위반일이 없으면 ES는 정의되지 않음 (0.0으로 두면 오해 소지)
        return {
            'tail_error': None,
            'mean_actual_es': None,
            'mean_sim_es': None,
            'n_violations': 0
        }
    
    # 위반일에서의 실제 수익률 평균
    mean_actual_es = np.mean(actual_returns[E])
    # 위반일에서의 시뮬레이션 ES 평균
    mean_sim_es = np.mean(es_t[E])
    # tail_error = mean(r_t|E) - mean(ES_t|E)
    tail_error = mean_actual_es - mean_sim_es
    
    return {
        'tail_error': float(tail_error),
        'mean_actual_es': float(mean_actual_es),
        'mean_sim_es': float(mean_sim_es),
        'n_violations': int(len(E))
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
    
    # 전체 유효성 판단 (PIT-KS, Kupiec p-value > 0.05)
    is_valid = (pit_ks_result['ks_pvalue'] > 0.05 and kupiec_result['pvalue'] > 0.05)
    
    return {
        'pit_ks': pit_ks_result,
        'kupiec': kupiec_result,
        'es': es_result,
        'is_valid': is_valid
    }
