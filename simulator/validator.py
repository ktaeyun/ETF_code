"""
시뮬레이션 결과 검증 모듈
(i) 수익률 분포 꼬리, (ii) 변동성 군집(ACF of r^2), (iii) 점프 간격 분포(지수근사) 비교
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


def validate_simulation(simulated_returns, actual_returns, simulated_jump_times=None, actual_jump_times=None):
    """
    시뮬레이션 결과 검증
    
    Args:
        simulated_returns: 시뮬레이션된 수익률
        actual_returns: 실제 수익률
        simulated_jump_times: 시뮬레이션된 점프 시점 (선택)
        actual_jump_times: 실제 점프 시점 (선택)
    
    Returns:
        dict: 검증 결과
    """
    results = {}
    
    # (i) 수익률 분포 꼬리 비교
    tail_results = compare_tail_distribution(simulated_returns, actual_returns)
    results['tail_distribution'] = tail_results
    
    # (ii) 변동성 군집 (ACF of r^2)
    volatility_clustering = compare_volatility_clustering(simulated_returns, actual_returns)
    results['volatility_clustering'] = volatility_clustering
    
    # (iii) 점프 간격 분포 (지수근사)
    if simulated_jump_times is not None and actual_jump_times is not None:
        jump_intervals = compare_jump_intervals(simulated_jump_times, actual_jump_times)
        results['jump_intervals'] = jump_intervals
    
    return results


def compare_tail_distribution(simulated, actual, quantiles=[0.95, 0.975, 0.99, 0.995]):
    """
    수익률 분포 꼬리 비교
    
    Args:
        simulated: 시뮬레이션된 수익률
        actual: 실제 수익률
        quantiles: 비교할 분위수 리스트
    
    Returns:
        dict: 꼬리 분포 비교 결과
    """
    sim_values = np.array(simulated)
    actual_values = np.array(actual)
    
    results = {}
    
    for q in quantiles:
        sim_q = np.quantile(np.abs(sim_values), q)
        actual_q = np.quantile(np.abs(actual_values), q)
        
        results[f'quantile_{q}'] = {
            'simulated': sim_q,
            'actual': actual_q,
            'ratio': sim_q / actual_q if actual_q != 0 else np.nan
        }
    
    # 전체 통계
    results['statistics'] = {
        'simulated': {
            'mean': np.mean(sim_values),
            'std': np.std(sim_values),
            'skew': stats.skew(sim_values),
            'kurtosis': stats.kurtosis(sim_values)
        },
        'actual': {
            'mean': np.mean(actual_values),
            'std': np.std(actual_values),
            'skew': stats.skew(actual_values),
            'kurtosis': stats.kurtosis(actual_values)
        }
    }
    
    print(f"\n수익률 분포 꼬리 비교:")
    print(f"  실제: mean={results['statistics']['actual']['mean']:.6f}, std={results['statistics']['actual']['std']:.6f}")
    print(f"  시뮬: mean={results['statistics']['simulated']['mean']:.6f}, std={results['statistics']['simulated']['std']:.6f}")
    print(f"  실제: skew={results['statistics']['actual']['skew']:.4f}, kurtosis={results['statistics']['actual']['kurtosis']:.4f}")
    print(f"  시뮬: skew={results['statistics']['simulated']['skew']:.4f}, kurtosis={results['statistics']['simulated']['kurtosis']:.4f}")
    
    return results


def compare_volatility_clustering(simulated, actual, max_lag=20):
    """
    변동성 군집 비교 (ACF of r^2)
    
    Args:
        simulated: 시뮬레이션된 수익률
        actual: 실제 수익률
        max_lag: 최대 lag
    
    Returns:
        dict: 변동성 군집 비교 결과
    """
    sim_squared = np.array(simulated) ** 2
    actual_squared = np.array(actual) ** 2
    
    # ACF 계산
    sim_acf = acf(sim_squared, nlags=max_lag, fft=True)
    actual_acf = acf(actual_squared, nlags=max_lag, fft=True)
    
    results = {
        'simulated_acf': sim_acf.tolist(),
        'actual_acf': actual_acf.tolist(),
        'max_lag': max_lag,
        'correlation': np.corrcoef(sim_acf, actual_acf)[0, 1]
    }
    
    print(f"\n변동성 군집 비교 (ACF of r^2):")
    print(f"  Lag 1: 실제={actual_acf[1]:.4f}, 시뮬={sim_acf[1]:.4f}")
    print(f"  Lag 5: 실제={actual_acf[5]:.4f}, 시뮬={sim_acf[5]:.4f}")
    print(f"  Lag 10: 실제={actual_acf[10]:.4f}, 시뮬={sim_acf[10]:.4f}")
    print(f"  ACF 상관계수: {results['correlation']:.4f}")
    
    return results


def compare_jump_intervals(simulated_jump_times, actual_jump_times):
    """
    점프 간격 분포 비교 (지수근사)
    
    Args:
        simulated_jump_times: 시뮬레이션된 점프 시점
        actual_jump_times: 실제 점프 시점
    
    Returns:
        dict: 점프 간격 비교 결과
    """
    # 점프 간격 계산
    sim_intervals = np.diff(np.sort(simulated_jump_times))
    actual_intervals = np.diff(np.sort(actual_jump_times))
    
    if len(sim_intervals) == 0 or len(actual_intervals) == 0:
        return None
    
    # 지수분포 파라미터 추정 (MLE: lambda = 1 / mean)
    sim_lambda = 1 / np.mean(sim_intervals) if np.mean(sim_intervals) > 0 else 0
    actual_lambda = 1 / np.mean(actual_intervals) if np.mean(actual_intervals) > 0 else 0
    
    # Kolmogorov-Smirnov 검정 (지수분포 적합도)
    from scipy.stats import kstest, expon
    
    sim_ks_stat, sim_ks_pvalue = kstest(
        sim_intervals,
        lambda x: expon.cdf(x, scale=1/sim_lambda) if sim_lambda > 0 else 0
    )
    
    actual_ks_stat, actual_ks_pvalue = kstest(
        actual_intervals,
        lambda x: expon.cdf(x, scale=1/actual_lambda) if actual_lambda > 0 else 0
    )
    
    results = {
        'simulated': {
            'mean_interval': np.mean(sim_intervals),
            'lambda': sim_lambda,
            'ks_statistic': sim_ks_stat,
            'ks_pvalue': sim_ks_pvalue
        },
        'actual': {
            'mean_interval': np.mean(actual_intervals),
            'lambda': actual_lambda,
            'ks_statistic': actual_ks_stat,
            'ks_pvalue': actual_ks_pvalue
        }
    }
    
    print(f"\n점프 간격 분포 비교:")
    print(f"  실제: 평균 간격={np.mean(actual_intervals):.2f}, λ={actual_lambda:.6f}")
    print(f"  시뮬: 평균 간격={np.mean(sim_intervals):.2f}, λ={sim_lambda:.6f}")
    print(f"  실제 KS 검정: 통계량={actual_ks_stat:.4f}, p-value={actual_ks_pvalue:.4f}")
    print(f"  시뮬 KS 검정: 통계량={sim_ks_stat:.4f}, p-value={sim_ks_pvalue:.4f}")
    
    return results
