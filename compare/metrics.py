"""
NAV 시뮬레이터 검증 지표 모듈
금융공학 연구용 평가 지표 구현
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.stats import kstest, anderson_ksamp, ks_2samp, chi2
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
import warnings


def dtw_distance(series1, series2):
    """
    Dynamic Time Warping (DTW) 거리 계산
    
    Args:
        series1: 시계열 1
        series2: 시계열 2
    
    Returns:
        float: DTW 거리 (정규화된 값)
    """
    s1 = np.array(series1)
    s2 = np.array(series2)
    n, m = len(s1), len(s2)
    
    # 스케일 정규화 (0~1 범위로)
    if np.max(s1) - np.min(s1) > 0:
        s1 = (s1 - np.min(s1)) / (np.max(s1) - np.min(s1))
    if np.max(s2) - np.min(s2) > 0:
        s2 = (s2 - np.min(s2)) / (np.max(s2) - np.min(s2))
    
    # DTW 행렬 초기화
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # DTW 계산
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    # 정규화된 DTW 거리 반환
    return dtw_matrix[n, m] / max(n, m)


def wmcr_price(actual_price, simulated_paths, bands=[0.2, 0.3, 0.4, 0.5, 0.6], weights=[5, 4, 3, 2, 1]):
    """
    Weighted Multi-band Capture Rate (WMCR) Price
    논문 정의: 시뮬레이션 경로들의 밴드 내에 실제 가격이 들어가는 비율
    
    Args:
        actual_price: 실제 가격 시계열 (T,)
        simulated_paths: 시뮬레이션 경로들 (N x T 배열)
        bands: 퍼센트 밴드 리스트 (예: [0.2, 0.3, 0.4, 0.5, 0.6] = 20%, 30%, ...)
        weights: 각 밴드에 대한 가중치 리스트
    
    Returns:
        float: WMCR Price 값 (높을수록 좋음, 0~1)
    """
    actual_price = np.array(actual_price)
    simulated_paths = np.array(simulated_paths)
    
    T = len(actual_price)
    total_score = 0.0
    total_weight = 0.0
    
    for pb, wb in zip(bands, weights):
        capture_count = 0
        for t in range(T):
            # 시뮬레이션 경로들의 min, max
            min_sim = np.min(simulated_paths[:, t])
            max_sim = np.max(simulated_paths[:, t])
            midpoint = (min_sim + max_sim) / 2
            
            # 밴드 범위
            lower_bound = midpoint * (1 - pb)
            upper_bound = midpoint * (1 + pb)
            
            # 실제 가격이 밴드 내에 있는지 확인
            if lower_bound <= actual_price[t] <= upper_bound:
                capture_count += 1
        
        capture_rate = capture_count / T
        total_score += wb * capture_rate
        total_weight += wb
    
    return total_score / total_weight if total_weight > 0 else 0.0


def wmcr_volatility(actual_vol, simulated_vol_paths, bands=[0.2, 0.3, 0.4, 0.5, 0.6], weights=[5, 4, 3, 2, 1]):
    """
    Weighted Multi-band Capture Rate (WMCR) Volatility
    논문 정의: 시뮬레이션 변동성 경로들의 밴드 내에 실제 변동성이 들어가는 비율
    
    Args:
        actual_vol: 실제 변동성 시계열 (T,)
        simulated_vol_paths: 시뮬레이션 변동성 경로들 (N x T 배열)
        bands: 퍼센트 밴드 리스트
        weights: 각 밴드에 대한 가중치 리스트
    
    Returns:
        float: WMCR Volatility 값 (높을수록 좋음, 0~1)
    """
    return wmcr_price(actual_vol, simulated_vol_paths, bands=bands, weights=weights)


def wmcr(series1, series2, benchmark_series=None, window_size=None):
    """
    Weighted Mean Cross-Correlation Rate (WMCR) - 구버전 (호환성 유지)
    기준 대비 스코어: (실제 상관계수 - 기준 상관계수)
    
    Args:
        series1: 시계열 1 (실제 데이터)
        series2: 시계열 2 (시뮬레이션 데이터)
        benchmark_series: 기준 시계열 (None이면 series1의 자기상관계수 사용)
        window_size: 윈도우 크기 (None이면 전체 길이의 10%)
    
    Returns:
        float: WMCR 값 (기준 대비 스코어, 양수=기준 이상, 음수=기준 미만)
    """
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    if window_size is None:
        window_size = max(10, len(s1) // 10)
    
    # 실제 상관계수 계산 (series1 vs series2)
    actual_correlations = []
    weights = []
    
    for i in range(len(s1) - window_size + 1):
        window1 = s1[i:i+window_size]
        window2 = s2[i:i+window_size]
        
        if np.std(window1) > 0 and np.std(window2) > 0:
            corr = np.corrcoef(window1, window2)[0, 1]
            if not np.isnan(corr):
                actual_correlations.append(corr)
                # 가중치: 최근 윈도우에 더 높은 가중치
                weights.append(i + 1)
    
    if len(actual_correlations) == 0:
        return 0.0
    
    weights = np.array(weights)
    weights = weights / weights.sum()
    actual_wmcr = np.average(actual_correlations, weights=weights)
    
    # 기준 상관계수 계산
    # 기준값: 실제 데이터(series1)의 자기상관계수를 기준으로 사용
    # 하지만 시뮬레이션 평가에서는 기준값을 더 낮게 설정하는 것이 합리적
    # (실제 데이터와 시뮬레이션 데이터 간의 상관계수가 자기상관계수보다 낮을 수 있음)
    from statsmodels.tsa.stattools import acf
    
    if benchmark_series is not None:
        # 기준 시계열과의 상관계수
        benchmark_correlations = []
        benchmark_weights = []
        benchmark_s = np.array(benchmark_series)
        
        for i in range(len(s1) - window_size + 1):
            window1 = s1[i:i+window_size]
            window_bench = benchmark_s[i:i+window_size]
            
            if np.std(window1) > 0 and np.std(window_bench) > 0:
                corr = np.corrcoef(window1, window_bench)[0, 1]
                if not np.isnan(corr):
                    benchmark_correlations.append(corr)
                    benchmark_weights.append(i + 1)
        
        if len(benchmark_correlations) > 0:
            benchmark_weights = np.array(benchmark_weights)
            benchmark_weights = benchmark_weights / benchmark_weights.sum()
            benchmark_wmcr = np.average(benchmark_correlations, weights=benchmark_weights)
        else:
            # Fallback: series1의 자기상관계수 사용
            autocorr = acf(s1, nlags=min(5, len(s1)//4), fft=True)
            autocorr_mean = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0.5
            # 기준값을 자기상관계수의 일정 비율로 조정 (예: 0.7배)
            benchmark_wmcr = autocorr_mean * 0.7
    else:
        # 기준값: series1의 자기상관계수 (lag 1-5 평균)
        # 하지만 시뮬레이션 평가를 위해 기준값을 조정
        autocorr = acf(s1, nlags=min(5, len(s1)//4), fft=True)
        autocorr_mean = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0.5
        
        # 기준값 설정 옵션:
        # 1. 자기상관계수 그대로 사용 (너무 높을 수 있음)
        # 2. 자기상관계수의 일정 비율 사용 (예: 0.6~0.8)
        # 3. 고정 기준값 사용 (예: 0.3~0.5)
        
        # 여기서는 자기상관계수의 0.6배를 기준으로 사용
        # (실제 데이터와 시뮬레이션 간 상관계수가 자기상관계수보다 낮을 수 있음을 고려)
        benchmark_wmcr = autocorr_mean * 0.6
    
    # WMCR = 실제 상관계수 - 기준 상관계수
    wmcr_score = actual_wmcr - benchmark_wmcr
    
    # 디버깅 정보 (필요시 주석 해제)
    # print(f"    WMCR 디버깅: actual={actual_wmcr:.4f}, benchmark={benchmark_wmcr:.4f}, score={wmcr_score:.4f}")
    
    return wmcr_score


def pmc(series1, series2):
    """
    Path Momentum Consistency (PMC)
    모멘텀 방향의 일치성 측정
    
    Args:
        series1: 시계열 1
        series2: 시계열 2
    
    Returns:
        float: PMC 값 (0~1, 1에 가까울수록 일치)
    """
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    # 모멘텀 (변화율) 계산
    momentum1 = np.diff(s1)
    momentum2 = np.diff(s2)
    
    # 부호 일치 비율
    sign_match = np.mean(np.sign(momentum1) == np.sign(momentum2))
    
    # 크기 상관관계
    if np.std(momentum1) > 0 and np.std(momentum2) > 0:
        size_corr = np.corrcoef(momentum1, momentum2)[0, 1]
        if np.isnan(size_corr):
            size_corr = 0
    else:
        size_corr = 0
    
    # PMC: 부호 일치와 크기 상관관계의 가중 평균
    pmc_value = 0.5 * sign_match + 0.5 * (size_corr + 1) / 2
    
    return pmc_value


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


def rvr(actual_returns, simulated_returns, window=20):
    """
    Realized Volatility Regression (RVR)
    R^2와 β1의 1 근접성 측정
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        window: Realized Volatility 계산 윈도우
    
    Returns:
        dict: {'r_squared': R^2, 'beta1': β1, 'beta1_proximity': |β1 - 1|}
    """
    rv_actual = realized_volatility(actual_returns, window)
    rv_sim = realized_volatility(simulated_returns, window)
    
    # 유효한 값만 선택
    valid_mask = ~(np.isnan(rv_actual) | np.isnan(rv_sim))
    rv_actual_valid = rv_actual[valid_mask]
    rv_sim_valid = rv_sim[valid_mask]
    
    if len(rv_actual_valid) < 10:
        return {'r_squared': 0.0, 'beta1': 0.0, 'beta1_proximity': 1.0}
    
    # 선형 회귀: RV_actual = β0 + β1 * RV_sim
    try:
        from statsmodels.api import add_constant
        X = add_constant(rv_sim_valid)
        model = OLS(rv_actual_valid, X).fit()
        r_squared = model.rsquared
        beta1 = model.params[1] if len(model.params) > 1 else model.params[0] if len(model.params) > 0 else 0.0
        beta1_proximity = abs(beta1 - 1.0)
    except Exception as e:
        r_squared = 0.0
        beta1 = 0.0
        beta1_proximity = 1.0
    
    return {
        'r_squared': r_squared,
        'beta1': beta1,
        'beta1_proximity': beta1_proximity
    }


def vvs(actual_vol, simulated_vol_paths):
    """
    Volatility of Volatility Similarity (VVS)
    논문 정의: 변동성의 변동성(VoV) 유사성
    
    Args:
        actual_vol: 실제 변동성 시계열 (T,)
        simulated_vol_paths: 시뮬레이션 변동성 경로들 (N x T 배열)
    
    Returns:
        float: VVS 값 (0~1, 1에 가까울수록 좋음)
    """
    actual_vol = np.array(actual_vol)
    simulated_vol_paths = np.array(simulated_vol_paths)
    
    # 1차원 배열인 경우 처리
    if simulated_vol_paths.ndim == 1:
        simulated_vol_paths = simulated_vol_paths.reshape(1, -1)
    
    # 실제 변동성의 VoV 계산
    if len(actual_vol) < 2:
        return 0.0
    
    delta_actual = np.abs(np.diff(actual_vol))
    mean_delta_actual = np.mean(delta_actual)
    vov_actual = np.sqrt(np.mean((delta_actual - mean_delta_actual)**2))
    
    # 시뮬레이션 변동성의 평균 VoV 계산
    vov_sim_list = []
    for vol_path in simulated_vol_paths:
        if len(vol_path) < 2:
            continue
        delta_sim = np.abs(np.diff(vol_path))
        mean_delta_sim = np.mean(delta_sim)
        vov_sim = np.sqrt(np.mean((delta_sim - mean_delta_sim)**2))
        vov_sim_list.append(vov_sim)
    
    if len(vov_sim_list) == 0:
        return 0.0
    
    vov_sim_avg = np.mean(vov_sim_list)
    
    # VVS = 1 - |VoV_sim - VoV_actual| / (VoV_sim + VoV_actual)
    if vov_sim_avg + vov_actual == 0:
        return 0.0
    
    vvs_value = 1 - abs(vov_sim_avg - vov_actual) / (vov_sim_avg + vov_actual)
    return max(0.0, min(1.0, vvs_value))


def vpr(actual_vol, simulated_vol_paths):
    """
    Volatility Persistence Ratio (VPR)
    논문 정의: 변동성의 autocovariance 비율
    
    Args:
        actual_vol: 실제 변동성 시계열 (T,)
        simulated_vol_paths: 시뮬레이션 변동성 경로들 (N x T 배열)
    
    Returns:
        float: VPR 값 (1에 가까울수록 좋음)
    """
    actual_vol = np.array(actual_vol)
    simulated_vol_paths = np.array(simulated_vol_paths)
    
    # 1차원 배열인 경우 처리
    if simulated_vol_paths.ndim == 1:
        simulated_vol_paths = simulated_vol_paths.reshape(1, -1)
    
    T = len(actual_vol)
    
    if T < 2:
        return 0.0
    
    # 실제 변동성의 autocovariance
    mean_actual = np.mean(actual_vol)
    autocov_actual = 0.0
    for i in range(T - 1):
        autocov_actual += (actual_vol[i+1] - mean_actual) * (actual_vol[i] - mean_actual)
    autocov_actual = autocov_actual / (T - 1)
    
    # 시뮬레이션 변동성의 평균 autocovariance
    autocov_sim_list = []
    for vol_path in simulated_vol_paths:
        if len(vol_path) != T:
            continue
        mean_sim = np.mean(vol_path)
        autocov_sim = 0.0
        for i in range(T - 1):
            autocov_sim += (vol_path[i+1] - mean_sim) * (vol_path[i] - mean_sim)
        autocov_sim = autocov_sim / (T - 1)
        autocov_sim_list.append(autocov_sim)
    
    if len(autocov_sim_list) == 0:
        return 0.0
    
    autocov_sim_avg = np.mean(autocov_sim_list)
    
    # VPR = autocov_sim / autocov_actual
    if autocov_actual == 0:
        return 0.0
    
    return autocov_sim_avg / autocov_actual


def vjc(actual_vol, simulated_vol_paths):
    """
    Volatility Jump Capture (VJC)
    논문 정의: 실제 변동성 점프를 시뮬레이션 경로들이 포착하는 비율
    
    Args:
        actual_vol: 실제 변동성 시계열 (T,)
        simulated_vol_paths: 시뮬레이션 변동성 경로들 (N x T 배열)
    
    Returns:
        float: VJC 값 (높을수록 좋음, 0~1)
    """
    actual_vol = np.array(actual_vol)
    simulated_vol_paths = np.array(simulated_vol_paths)
    
    # 1차원 배열인 경우 처리
    if simulated_vol_paths.ndim == 1:
        simulated_vol_paths = simulated_vol_paths.reshape(1, -1)
    
    T = len(actual_vol)
    
    if T < 2:
        return 0.0
    
    # 실제 변동성 점프 감지
    delta_actual = np.abs(np.diff(actual_vol))
    if len(delta_actual) == 0:
        return 0.0
    
    threshold_actual = np.quantile(delta_actual, 0.95)  # 상위 5%를 점프로 간주
    actual_jump_times = np.where(delta_actual > threshold_actual)[0]
    N_jumps = len(actual_jump_times)
    
    if N_jumps == 0:
        return 0.0
    
    # 각 시뮬레이션 경로에서 점프 포착 확인
    total_captured = 0.0
    for vol_path in simulated_vol_paths:
        if len(vol_path) != T:
            continue
        delta_sim = np.abs(np.diff(vol_path))
        if len(delta_sim) == 0:
            continue
        threshold_sim = np.quantile(delta_sim, 0.95)
        
        for tj in actual_jump_times:
            if tj < len(delta_sim) and delta_sim[tj] > threshold_sim:
                total_captured += 1
    
    # VJC = 평균 포착 비율
    n_paths = len(simulated_vol_paths)
    if n_paths == 0:
        return 0.0
    
    vjc_value = total_captured / (n_paths * N_jumps)
    return vjc_value


def twad(actual_returns, simulated_returns, tail_weight=0.1):
    """
    Tail-Weighted Anderson-Darling (TWAD)
    꼬리 가중 Anderson-Darling 검정
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        tail_weight: 꼬리 가중치
    
    Returns:
        float: TWAD 통계량 (작을수록 유사)
    """
    actual_values = np.array(actual_returns)
    sim_values = np.array(simulated_returns)
    
    # 꼬리 영역 정의 (상위/하위 tail_weight%)
    lower_tail = np.quantile(actual_values, tail_weight)
    upper_tail = np.quantile(actual_values, 1 - tail_weight)
    
    # 꼬리 영역 데이터만 선택
    actual_tail = actual_values[(actual_values <= lower_tail) | (actual_values >= upper_tail)]
    sim_tail = sim_values[(sim_values <= lower_tail) | (sim_values >= upper_tail)]
    
    if len(actual_tail) < 5 or len(sim_tail) < 5:
        return np.inf
    
    # Anderson-Darling 검정 (간단한 근사)
    # 실제로는 더 정교한 구현 필요하지만, 여기서는 간단히 처리
    try:
        # 두 샘플의 경험적 분포 함수 비교
        all_values = np.concatenate([actual_tail, sim_tail])
        sorted_values = np.sort(all_values)
        
        n1, n2 = len(actual_tail), len(sim_tail)
        n = n1 + n2
        
        ad_stat = 0.0
        for i, val in enumerate(sorted_values):
            f1 = np.sum(actual_tail <= val) / n1
            f2 = np.sum(sim_tail <= val) / n2
            
            if i > 0 and i < len(sorted_values) - 1:
                weight = 1.0 / (i * (n - i))
                ad_stat += weight * (f1 - f2)**2
        
        return ad_stat
    except:
        return np.inf


def ks_test(actual_returns, simulated_returns):
    """
    Kolmogorov-Smirnov 검정
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
    
    Returns:
        dict: {'statistic': KS 통계량, 'pvalue': p-value}
    """
    actual_values = np.array(actual_returns)
    sim_values = np.array(simulated_returns)
    
    try:
        # 두 표본의 경험적 분포 함수 비교
        from scipy.stats import ks_2samp
        ks_stat, pvalue = ks_2samp(actual_values, sim_values)
    except:
        # Fallback: 간단한 방법
        try:
            ks_stat, pvalue = kstest(actual_values, lambda x: np.mean(sim_values <= x))
        except:
            ks_stat = 1.0
            pvalue = 0.0
    
    return {
        'statistic': ks_stat,
        'pvalue': pvalue
    }


def distribution_moments(actual_returns, simulated_returns):
    """
    분포 모멘트 근접성 (mean, median, std)
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
    
    Returns:
        dict: {'mean_proximity': mean 근접성, 'median_proximity': median 근접성, 'std_proximity': std 근접성}
    """
    actual_values = np.array(actual_returns)
    sim_values = np.array(simulated_returns)
    
    actual_mean = np.mean(actual_values)
    sim_mean = np.mean(sim_values)
    mean_proximity = 1 - abs(actual_mean - sim_mean) / (abs(actual_mean) + 1e-10)
    
    actual_median = np.median(actual_values)
    sim_median = np.median(sim_values)
    median_proximity = 1 - abs(actual_median - sim_median) / (abs(actual_median) + 1e-10)
    
    actual_std = np.std(actual_values)
    sim_std = np.std(sim_values)
    std_proximity = 1 - abs(actual_std - sim_std) / (actual_std + 1e-10)
    
    return {
        'mean_proximity': mean_proximity,
        'median_proximity': median_proximity,
        'std_proximity': std_proximity,
        'actual_mean': actual_mean,
        'sim_mean': sim_mean,
        'actual_median': actual_median,
        'sim_median': sim_median,
        'actual_std': actual_std,
        'sim_std': sim_std
    }


def var_es_comparison(actual_returns, simulated_returns, alpha=0.05):
    """
    VaR(5%) 및 ES(5%) 근접성 비교
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        alpha: 유의수준 (기본값: 0.05 = 5%)
    
    Returns:
        dict: VaR 및 ES 비교 결과
    """
    actual_values = np.array(actual_returns)
    sim_values = np.array(simulated_returns)
    
    # VaR 계산 (하위 α 분위수)
    actual_var = np.quantile(actual_values, alpha)
    sim_var = np.quantile(sim_values, alpha)
    var_proximity = 1 - abs(actual_var - sim_var) / (abs(actual_var) + 1e-10)
    
    # ES 계산 (Conditional VaR, 평균)
    actual_es = np.mean(actual_values[actual_values <= actual_var])
    sim_es = np.mean(sim_values[sim_values <= sim_var])
    es_proximity = 1 - abs(actual_es - sim_es) / (abs(actual_es) + 1e-10)
    
    return {
        'var_proximity': var_proximity,
        'es_proximity': es_proximity,
        'actual_var': actual_var,
        'sim_var': sim_var,
        'actual_es': actual_es,
        'sim_es': sim_es
    }


def calculate_all_metrics(actual_nav, simulated_nav, actual_returns, simulated_returns,
                          monte_carlo_nav_paths=None, monte_carlo_returns_paths=None):
    """
    모든 검증 지표 계산 (논문 정의에 맞춤)
    
    Args:
        actual_nav: 실제 NAV 시계열
        simulated_nav: 시뮬레이션된 NAV 시계열 (대표 경로)
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률 (대표 경로)
        monte_carlo_nav_paths: 몬테카를로 시뮬레이션 NAV 경로들 (N x T 배열, 선택)
        monte_carlo_returns_paths: 몬테카를로 시뮬레이션 수익률 경로들 (N x T 배열, 선택)
    
    Returns:
        dict: 모든 검증 지표 결과
    """
    metrics = {}
    
    # Price Path Metrics
    print("  [Price Path Metrics] 계산 중...")
    metrics['dtw_price'] = dtw_distance(actual_nav, simulated_nav)
    
    # WMCR Price: 몬테카를로 경로가 있으면 논문 정의 사용
    if monte_carlo_nav_paths is not None and len(monte_carlo_nav_paths) > 0:
        if isinstance(monte_carlo_nav_paths, list):
            monte_carlo_nav_paths = np.array(monte_carlo_nav_paths)
        metrics['wmcr_price'] = wmcr_price(actual_nav, monte_carlo_nav_paths)
    else:
        # 단일 경로인 경우 구버전 사용
        metrics['wmcr_price'] = wmcr(actual_nav, simulated_nav, benchmark_series=None)
    
    metrics['pmc'] = pmc(actual_nav, simulated_nav)
    
    # Volatility Metrics
    print("  [Volatility Metrics] 계산 중...")
    actual_rv = realized_volatility(actual_returns)
    sim_rv = realized_volatility(simulated_returns)
    metrics['dtw_vol'] = dtw_distance(actual_rv, sim_rv)
    
    # WMCR Vol: 몬테카를로 경로가 있으면 논문 정의 사용
    if monte_carlo_returns_paths is not None and len(monte_carlo_returns_paths) > 0:
        if isinstance(monte_carlo_returns_paths, list):
            monte_carlo_returns_paths = np.array(monte_carlo_returns_paths)
        # 시뮬레이션 수익률에서 변동성 경로 생성
        sim_vol_paths = np.array([realized_volatility(ret) for ret in monte_carlo_returns_paths])
        metrics['wmcr_vol'] = wmcr_volatility(actual_rv, sim_vol_paths)
    else:
        # 단일 경로인 경우 구버전 사용
        metrics['wmcr_vol'] = wmcr(actual_rv, sim_rv, benchmark_series=None)
    
    metrics['rvr'] = rvr(actual_returns, simulated_returns)
    
    # VVS, VPR, VJC: 몬테카를로 경로 필요
    if monte_carlo_returns_paths is not None and len(monte_carlo_returns_paths) > 0:
        if isinstance(monte_carlo_returns_paths, list):
            monte_carlo_returns_paths = np.array(monte_carlo_returns_paths)
        sim_vol_paths = np.array([realized_volatility(ret) for ret in monte_carlo_returns_paths])
        metrics['vvs'] = vvs(actual_rv, sim_vol_paths)
        metrics['vpr'] = vpr(actual_rv, sim_vol_paths)
        metrics['vjc'] = vjc(actual_rv, sim_vol_paths)
    else:
        # 단일 경로인 경우 구버전 사용
        metrics['vvs'] = vvs(actual_returns, simulated_returns)
        metrics['vpr'] = vpr(actual_returns, simulated_returns)
        metrics['vjc'] = vjc(actual_returns, simulated_returns)
    
    # Return Distribution & Risk
    print("  [Return Distribution & Risk Metrics] 계산 중...")
    metrics['twad'] = twad(actual_returns, simulated_returns)
    metrics['ks'] = ks_test(actual_returns, simulated_returns)
    metrics['distribution_moments'] = distribution_moments(actual_returns, simulated_returns)
    metrics['var_es'] = var_es_comparison(actual_returns, simulated_returns)
    
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
    x = np.sum(I_t)  # 초과 횟수
    rate = x / T  # 초과율
    
    if x == 0:
        # 초과가 없으면 p-value = 1
        lr_uc = 0.0
        pvalue = 1.0
    elif x == T:
        # 모두 초과면 p-value = 0
        lr_uc = np.inf
        pvalue = 0.0
    else:
        # LLR = -2 * (log L(H0) - log L(H1))
        # H0: 실제 초과율 = alpha
        # H1: 실제 초과율 != alpha
        lr_uc = -2 * (stats.binom.logpmf(x, T, alpha) - stats.binom.logpmf(x, T, rate))
        # 카이제곱 분포로 p-value 계산
        pvalue = 1 - chi2.cdf(lr_uc, df=1)
    
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
        # 위반일이 없으면 tail_error = 0 (완벽)
        return {
            'tail_error': 0.0,
            'mean_actual_es': 0.0,
            'mean_sim_es': 0.0,
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
    통계적 검정 수행 (PIT-KS, VaR-Kupiec, ES)
    
    Args:
        actual_returns: 실제 수익률 시계열 (T,)
        simulated_returns_paths: 몬테카를로 시뮬레이션 수익률 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'pit_ks': {...}, 'kupiec': {...}, 'es': {...}}
    """
    actual_returns = np.array(actual_returns)
    simulated_returns_paths = np.array(simulated_returns_paths)
    
    # 1. PIT-KS 검정
    pit_ks_result = pit_ks_test(actual_returns, simulated_returns_paths)
    
    # 2. VaR-Kupiec 검정
    kupiec_result = var_kupiec_test(actual_returns, simulated_returns_paths, alpha=alpha)
    
    # 3. ES 검정
    es_result = es_test(actual_returns, simulated_returns_paths, alpha=alpha)
    
    # 전체 유효성 판단 (모든 p-value > 0.05)
    is_valid = (pit_ks_result['ks_pvalue'] > 0.05 and 
                kupiec_result['pvalue'] > 0.05)
    
    return {
        'pit_ks': pit_ks_result,
        'kupiec': kupiec_result,
        'es': es_result,
        'is_valid': is_valid
    }
