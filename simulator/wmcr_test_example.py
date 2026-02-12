"""
WMCR 검정 사용 예제

이 예제는 WMCR 검정 통계량을 계산하고 p-value를 구하는 방법을 보여줍니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from simulator.wmcr_test import compute_wmcr, wmcr_pvalue_calibration


def example_basic_wmcr():
    """기본 WMCR 계산 예제"""
    print("=" * 60)
    print("예제 1: 기본 WMCR 계산")
    print("=" * 60)
    
    # 시뮬레이션 데이터 생성
    T = 100
    K = 3
    
    # 실제 관측값 (예: NAV 또는 수익률)
    np.random.seed(42)
    y_true = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    # 각 밴드의 하한/상한 (예: 시뮬레이션 경로의 분위수)
    # 밴드 1: 50% 커버리지 (25%~75% 분위수)
    # 밴드 2: 80% 커버리지 (10%~90% 분위수)
    # 밴드 3: 95% 커버리지 (2.5%~97.5% 분위수)
    bands_L = np.zeros((K, T))
    bands_U = np.zeros((K, T))
    
    # 시뮬레이션 경로 생성 (간단한 예제)
    n_sim = 1000
    sim_paths = np.zeros((n_sim, T))
    for i in range(n_sim):
        sim_paths[i] = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    # 각 시점에서 분위수 계산
    for t in range(T):
        sim_values_t = sim_paths[:, t]
        bands_L[0, t] = np.percentile(sim_values_t, 25)  # 50% 밴드 하한
        bands_U[0, t] = np.percentile(sim_values_t, 75)  # 50% 밴드 상한
        bands_L[1, t] = np.percentile(sim_values_t, 10)  # 80% 밴드 하한
        bands_U[1, t] = np.percentile(sim_values_t, 90)  # 80% 밴드 상한
        bands_L[2, t] = np.percentile(sim_values_t, 2.5)  # 95% 밴드 하한
        bands_U[2, t] = np.percentile(sim_values_t, 97.5)  # 95% 밴드 상한
    
    # 목표 커버리지 및 가중치
    p_targets = np.array([0.50, 0.80, 0.95])
    weights = np.array([3, 2, 1])  # 좁은 밴드에 더 높은 가중치
    
    # WMCR 계산
    wmcr = compute_wmcr(y_true, bands_L, bands_U, p_targets, weights, mode="absolute")
    
    print(f"관측값 길이: {len(y_true)}")
    print(f"밴드 수: {K}")
    print(f"목표 커버리지: {p_targets}")
    print(f"가중치: {weights}")
    print(f"WMCR 통계량: {wmcr:.4f}")
    print()


def example_wmcr_test():
    """WMCR 검정 (p-value 계산) 예제"""
    print("=" * 60)
    print("예제 2: WMCR 검정 및 p-value 계산")
    print("=" * 60)
    
    # 시뮬레이션 데이터 생성
    T = 100
    K = 3
    
    np.random.seed(42)
    y_true = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    # 밴드 구성 (간단한 예제)
    bands_L = np.zeros((K, T))
    bands_U = np.zeros((K, T))
    
    n_sim = 1000
    sim_paths = np.zeros((n_sim, T))
    for i in range(n_sim):
        sim_paths[i] = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    for t in range(T):
        sim_values_t = sim_paths[:, t]
        bands_L[0, t] = np.percentile(sim_values_t, 25)
        bands_U[0, t] = np.percentile(sim_values_t, 75)
        bands_L[1, t] = np.percentile(sim_values_t, 10)
        bands_U[1, t] = np.percentile(sim_values_t, 90)
        bands_L[2, t] = np.percentile(sim_values_t, 2.5)
        bands_U[2, t] = np.percentile(sim_values_t, 97.5)
    
    p_targets = np.array([0.50, 0.80, 0.95])
    weights = np.array([3, 2, 1])
    
    # WMCR 검정 실행
    result = wmcr_pvalue_calibration(
        y=y_true,
        bands_L=bands_L,
        bands_U=bands_U,
        p_targets=p_targets,
        weights=weights,
        B=500,  # 부트스트랩 반복 횟수 (실제로는 1000 이상 권장)
        block_len=10,  # 블록 길이
        method="circular_shift",  # 또는 "block_bootstrap"
        alternative="two-sided",
        mode="absolute",
        random_seed=42,
    )
    
    print(f"관측 WMCR: {result['wmcr_obs']:.4f}")
    print(f"p-value: {result['pvalue']:.4f}")
    print(f"각 밴드 캡처율: {[f'{c:.3f}' for c in result['coverage_rates']]}")
    print(f"목표 커버리지: {[f'{p:.2f}' for p in result['target_coverage']]}")
    print(f"방법: {result['method']}")
    print(f"대립가설: {result['alternative']}")
    print(f"부트스트랩 반복 횟수: {result['B']}")
    print()
    
    # 해석
    alpha = 0.05
    if result['pvalue'] < alpha:
        print(f"결론: p-value ({result['pvalue']:.4f}) < {alpha}, H0 기각")
        print("  → 모델이 목표 커버리지와 일치하지 않음 (캘리브레이션 실패)")
    else:
        print(f"결론: p-value ({result['pvalue']:.4f}) >= {alpha}, H0 채택")
        print("  → 모델이 목표 커버리지와 일치함 (캘리브레이션 성공)")
    print()


def example_different_modes():
    """다양한 정규화 모드 비교"""
    print("=" * 60)
    print("예제 3: 다양한 정규화 모드 비교")
    print("=" * 60)
    
    T = 100
    K = 2
    
    np.random.seed(42)
    y_true = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    bands_L = np.zeros((K, T))
    bands_U = np.zeros((K, T))
    
    n_sim = 1000
    sim_paths = np.zeros((n_sim, T))
    for i in range(n_sim):
        sim_paths[i] = np.cumsum(np.random.randn(T) * 0.01) + 100
    
    for t in range(T):
        sim_values_t = sim_paths[:, t]
        bands_L[0, t] = np.percentile(sim_values_t, 25)
        bands_U[0, t] = np.percentile(sim_values_t, 75)
        bands_L[1, t] = np.percentile(sim_values_t, 10)
        bands_U[1, t] = np.percentile(sim_values_t, 90)
    
    p_targets = np.array([0.50, 0.80])
    weights = np.array([1, 1])
    
    modes = ["absolute", "squared", "normalized"]
    for mode in modes:
        wmcr = compute_wmcr(y_true, bands_L, bands_U, p_targets, weights, mode=mode)
        print(f"{mode:12s} 모드: WMCR = {wmcr:.4f}")
    print()


if __name__ == "__main__":
    example_basic_wmcr()
    example_wmcr_test()
    example_different_modes()
