"""
괴리율 모델 검증 지표 모듈
PIT-KS, VaR-Kupiec, ES 검정 구현
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, chi2


def pit_ks_test(actual_changes, simulated_changes_paths):
    """
    PIT-KS 검정
    각 시점 t에서 z_t의 1-step predictive CDF F_hat_t를 만들고 u_t=F_hat_t(z_t)
    u_t가 U(0,1)인지 KS 검정
    
    Args:
        actual_changes: 실제 변화량 시계열 z_t (T,)
        simulated_changes_paths: 시뮬레이션 변화량 경로들 (N x T 배열)
    
    Returns:
        dict: {'ks_statistic': KS 통계량, 'ks_pvalue': p-value}
    """
    actual_changes = np.array(actual_changes)
    simulated_changes_paths = np.array(simulated_changes_paths)
    
    if simulated_changes_paths.ndim == 1:
        simulated_changes_paths = simulated_changes_paths.reshape(1, -1)
    
    T = len(actual_changes)
    N = len(simulated_changes_paths)
    
    # 각 시점 t에서 u_t = ECDF_t(z_t) 계산
    u_t = np.zeros(T)
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_changes_paths[:, t]
        # 경험적 CDF: ECDF_t(z_t) = (# of sim_values_t <= z_t) / N
        u_t[t] = np.mean(sim_values_t <= actual_changes[t])
    
    # u_t들이 U(0,1)인지 KS 검정
    ks_stat, ks_pvalue = kstest(u_t, 'uniform')
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue)
    }


def var_kupiec_test(actual_changes, simulated_changes_paths, alpha=0.05):
    """
    VaR-Kupiec 검정
    각 시점 t에서 VaR_t(alpha) = quantile(F_hat_t, alpha)
    I_t=1{z_t < VaR_t}; Kupiec LR_uc와 pvalue, 예외율 계산
    
    Args:
        actual_changes: 실제 변화량 시계열 z_t (T,)
        simulated_changes_paths: 시뮬레이션 변화량 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'lr_uc': LR 통계량, 'pvalue': p-value, 'exceedance_rate': 초과율, ...}
    """
    actual_changes = np.array(actual_changes)
    simulated_changes_paths = np.array(simulated_changes_paths)
    
    if simulated_changes_paths.ndim == 1:
        simulated_changes_paths = simulated_changes_paths.reshape(1, -1)
    
    T = len(actual_changes)
    
    # 각 시점 t에서 VaR_t 계산
    var_t = np.zeros(T)
    I_t = np.zeros(T, dtype=bool)
    
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_changes_paths[:, t]
        # VaR_t = quantile_i(z_hat^{(i)}_t, alpha)
        var_t[t] = np.quantile(sim_values_t, alpha)
        # I_t = 1{z_t < VaR_t}
        I_t[t] = actual_changes[t] < var_t[t]
    
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


def es_test(actual_changes, simulated_changes_paths, alpha=0.05):
    """
    ES (Expected Shortfall) 검정
    각 시점 t에서 ES_t(alpha)=E[z | z<=VaR_t] (예측분포 또는 시뮬 표본으로 계산)
    진단형 tail_error = mean(z_t|I_t=1) - mean(ES_t|I_t=1) (0에 가까울수록 좋음)
    
    Args:
        actual_changes: 실제 변화량 시계열 z_t (T,)
        simulated_changes_paths: 시뮬레이션 변화량 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'tail_error': tail_error, 'mean_actual_es': 실제 ES 평균, 'mean_sim_es': 시뮬레이션 ES 평균, 'n_violations': 위반 횟수}
    """
    actual_changes = np.array(actual_changes)
    simulated_changes_paths = np.array(simulated_changes_paths)
    
    if simulated_changes_paths.ndim == 1:
        simulated_changes_paths = simulated_changes_paths.reshape(1, -1)
    
    T = len(actual_changes)
    
    # 각 시점 t에서 VaR_t와 ES_t 계산
    var_t = np.zeros(T)
    es_t = np.zeros(T)
    I_t = np.zeros(T, dtype=bool)
    
    for t in range(T):
        # 시점 t에서의 시뮬레이션 값들
        sim_values_t = simulated_changes_paths[:, t]
        # VaR_t = quantile_i(z_hat^{(i)}_t, alpha)
        var_t[t] = np.quantile(sim_values_t, alpha)
        # ES_t = mean_i(z_hat^{(i)}_t | z_hat^{(i)}_t <= VaR_t)
        tail_samples = sim_values_t[sim_values_t <= var_t[t]]
        
        # 예외처리: VaR 이하 표본이 0개면 최소 1개 강제 포함
        if len(tail_samples) == 0:
            # 가장 작은 값 1개 포함
            tail_samples = np.array([np.min(sim_values_t)])
        
        es_t[t] = np.mean(tail_samples)
        # I_t = 1{z_t < VaR_t}
        I_t[t] = actual_changes[t] < var_t[t]
    
    # 위반일 E = {t: z_t < VaR_t}
    E = np.where(I_t)[0]
    
    if len(E) == 0:
        # 위반일이 없으면 tail_error = 0 (완벽)
        return {
            'tail_error': 0.0,
            'mean_actual_es': 0.0,
            'mean_sim_es': 0.0,
            'n_violations': 0
        }
    
    # 위반일에서의 실제 변화량 평균
    mean_actual_es = np.mean(actual_changes[E])
    # 위반일에서의 시뮬레이션 ES 평균
    mean_sim_es = np.mean(es_t[E])
    # tail_error = mean(z_t|E) - mean(ES_t|E)
    tail_error = mean_actual_es - mean_sim_es
    
    return {
        'tail_error': float(tail_error),
        'mean_actual_es': float(mean_actual_es),
        'mean_sim_es': float(mean_sim_es),
        'n_violations': int(len(E))
    }


def calculate_statistical_tests(actual_changes, simulated_changes_paths, alpha=0.05):
    """
    통계적 검정 수행 (PIT-KS, VaR-Kupiec, ES)
    
    Args:
        actual_changes: 실제 변화량 시계열 z_t (T,)
        simulated_changes_paths: 몬테카를로 시뮬레이션 변화량 경로들 (N x T 배열)
        alpha: VaR 유의수준 (기본값: 0.05)
    
    Returns:
        dict: {'pit_ks': {...}, 'kupiec': {...}, 'es': {...}, 'is_valid': bool}
    """
    actual_changes = np.array(actual_changes)
    simulated_changes_paths = np.array(simulated_changes_paths)
    
    # 1. PIT-KS 검정
    pit_ks_result = pit_ks_test(actual_changes, simulated_changes_paths)
    
    # 2. VaR-Kupiec 검정
    kupiec_result = var_kupiec_test(actual_changes, simulated_changes_paths, alpha=alpha)
    
    # 3. ES 검정
    es_result = es_test(actual_changes, simulated_changes_paths, alpha=alpha)
    
    # 전체 유효성 판단 (모든 p-value > 0.05)
    is_valid = (pit_ks_result['ks_pvalue'] > 0.05 and 
                kupiec_result['pvalue'] > 0.05)
    
    return {
        'pit_ks': pit_ks_result,
        'kupiec': kupiec_result,
        'es': es_result,
        'is_valid': is_valid
    }
