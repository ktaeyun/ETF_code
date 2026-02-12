"""
WMCR 이항비율 근사 검정 사용 예제
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from simulator.wmcr_test import wmcr_binomial_test


def example_wmcr_binomial_test():
    """WMCR 이항비율 근사 검정 예제"""
    print("=" * 80)
    print("예제: WMCR 이항비율 근사 검정")
    print("=" * 80)
    
    # 입력 데이터
    T = 343  # 테스트 길이
    p_targets = np.array([0.50, 0.80, 0.95])  # 목표 커버리지
    C_obs = np.array([0.3965, 0.8163, 0.8950])  # 관측 캡처율
    alpha = 0.05  # 유의수준
    
    # 검정 실행
    result = wmcr_binomial_test(
        T=T,
        p_targets=p_targets,
        C_obs=C_obs,
        alpha=alpha,
        verbose=True
    )
    
    # 결과 접근 예제
    print("\n결과 딕셔너리 키:", list(result.keys()))
    print(f"허용 가능한 밴드 수: {result['n_acceptable']}/{result['n_total']}")
    print(f"모든 밴드 허용 가능: {result['all_acceptable']}")
    print(f"문제가 있는 밴드: {result['problematic_bands']}")
    print(f"\n요약 텍스트:\n{result['summary_text']}")


def example_perfect_calibration():
    """완벽한 캘리브레이션 예제"""
    print("\n" + "=" * 80)
    print("예제: 완벽한 캘리브레이션 (모든 밴드 허용 가능)")
    print("=" * 80)
    
    T = 1000
    p_targets = np.array([0.50, 0.80, 0.95])
    # 목표 커버리지와 거의 일치하는 관측값
    C_obs = np.array([0.502, 0.798, 0.951])
    alpha = 0.05
    
    result = wmcr_binomial_test(
        T=T,
        p_targets=p_targets,
        C_obs=C_obs,
        alpha=alpha,
        verbose=True
    )


def example_poor_calibration():
    """캘리브레이션 실패 예제"""
    print("\n" + "=" * 80)
    print("예제: 캘리브레이션 실패 (여러 밴드에서 문제)")
    print("=" * 80)
    
    T = 200
    p_targets = np.array([0.50, 0.80, 0.95])
    # 목표 커버리지와 큰 차이
    C_obs = np.array([0.35, 0.90, 0.98])  # 과소 커버, 과대 커버, 과대 커버
    alpha = 0.05
    
    result = wmcr_binomial_test(
        T=T,
        p_targets=p_targets,
        C_obs=C_obs,
        alpha=alpha,
        verbose=True
    )


if __name__ == "__main__":
    example_wmcr_binomial_test()
    example_perfect_calibration()
    example_poor_calibration()
