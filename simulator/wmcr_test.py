"""
Weighted Multi-band Capture Rate (WMCR) 검정 통계량 및 p-value 계산

[방법론 설명 - 논문용]

Weighted Multi-band Capture Rate (WMCR)는 모델의 예측 구간 캘리브레이션을 평가하는 검정 통계량이다.
K개의 예측 구간 밴드 [L_{t,k}, U_{t,k}] (k=1,...,K)가 각 시점 t=1,...,T에 주어지고,
각 밴드 k는 목표 커버리지 p_k와 중요도 가중치 w_k를 가진다.
밴드 k의 관측 캡처율은 C_k = (1/T) * sum_{t=1}^T I_{t,k}로 정의되며,
여기서 I_{t,k} = 1{L_{t,k} <= y_t <= U_{t,k}}는 실제 관측값 y_t가 밴드 내에 있는지 여부를 나타낸다.

WMCR 통계량은 다음과 같이 정의된다:
  WMCR = (sum_{k=1}^K w_k * g(C_k, p_k)) / (sum_{k=1}^K w_k)

여기서 g(C_k, p_k) = 1 - |C_k - p_k|는 캡처율이 목표 커버리지에 가까울수록 큰 값을 반환하는 정규화 함수이다.
WMCR은 0과 1 사이의 값을 가지며, 1에 가까울수록 모든 밴드에서 목표 커버리지에 근접함을 의미한다.

귀무가설 H0는 "모든 밴드에서 C_k = p_k (캘리브레이션)"이고,
대립가설 H1은 "적어도 하나의 밴드에서 체계적 과대/과소 커버"이다.

시계열 의존성을 고려하여 p-value를 계산하기 위해, circular block shift 방법을 사용한다.
이 방법은 관측 시계열 y_t를 블록 단위로 순환 시프트하여 y_t와 밴드 [L_{t,k}, U_{t,k}] 간의
시간적 의존성을 깨뜨림으로써 H0 하에서의 귀무분포를 생성한다.
B회의 재표본을 통해 WMCR_null 분포를 구성하고,
p-value = (1 + #{WMCR_null >= WMCR_obs}) / (B+1)로 계산한다.
"""

import numpy as np
from typing import Union, List, Tuple, Literal, Dict
from scipy.stats import uniform, norm
import pandas as pd


def compute_wmcr(
    y: np.ndarray,
    bands_L: np.ndarray,
    bands_U: np.ndarray,
    p_targets: np.ndarray,
    weights: np.ndarray,
    mode: Literal["absolute", "squared", "normalized"] = "absolute"
) -> float:
    """
    WMCR 통계량 계산
    
    Args:
        y: 실제 관측값 시계열 (T,)
        bands_L: 각 밴드의 하한 (K x T) 또는 (T,) if K=1
        bands_U: 각 밴드의 상한 (K x T) 또는 (T,) if K=1
        p_targets: 각 밴드의 목표 커버리지 (K,)
        weights: 각 밴드의 가중치 (K,)
        mode: 정규화 함수 모드
            - "absolute": g(C_k, p_k) = 1 - |C_k - p_k|
            - "squared": g(C_k, p_k) = 1 - (C_k - p_k)^2 / max(p_k, 1-p_k)^2
            - "normalized": g(C_k, p_k) = 1 - |C_k - p_k| / max(p_k, 1-p_k)
    
    Returns:
        float: WMCR 통계량 (0~1, 높을수록 좋음)
    """
    y = np.asarray(y).flatten()
    bands_L = np.asarray(bands_L)
    bands_U = np.asarray(bands_U)
    p_targets = np.asarray(p_targets).flatten()
    weights = np.asarray(weights).flatten()
    
    T = len(y)
    K = len(p_targets)
    
    # 밴드 차원 확인 및 조정
    if bands_L.ndim == 1:
        bands_L = bands_L.reshape(1, -1)
    if bands_U.ndim == 1:
        bands_U = bands_U.reshape(1, -1)
    
    if bands_L.shape[0] != K or bands_U.shape[0] != K:
        raise ValueError(f"밴드 수 불일치: K={K}, bands_L.shape={bands_L.shape}, bands_U.shape={bands_U.shape}")
    if bands_L.shape[1] != T or bands_U.shape[1] != T:
        raise ValueError(f"시계열 길이 불일치: T={T}, bands_L.shape={bands_L.shape}, bands_U.shape={bands_U.shape}")
    
    # 각 밴드별 캡처율 계산
    C_k = np.zeros(K)
    for k in range(K):
        I_tk = (bands_L[k, :] <= y) & (y <= bands_U[k, :])
        C_k[k] = np.mean(I_tk)
    
    # 정규화 함수 g(C_k, p_k) 계산
    g_values = np.zeros(K)
    for k in range(K):
        diff = C_k[k] - p_targets[k]
        if mode == "absolute":
            g_values[k] = 1.0 - abs(diff)
        elif mode == "squared":
            denom = max(p_targets[k], 1 - p_targets[k]) ** 2
            g_values[k] = 1.0 - (diff ** 2) / denom if denom > 0 else 0.0
        elif mode == "normalized":
            denom = max(p_targets[k], 1 - p_targets[k])
            g_values[k] = 1.0 - abs(diff) / denom if denom > 0 else 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # WMCR = 가중 평균
    wmcr = np.sum(weights * g_values) / np.sum(weights) if np.sum(weights) > 0 else 0.0
    
    return float(wmcr)


def circular_block_shift(
    y: np.ndarray,
    block_len: int,
    shift: int = None
) -> np.ndarray:
    """
    Circular block shift: 시계열을 블록 단위로 순환 시프트
    
    Args:
        y: 원본 시계열 (T,)
        block_len: 블록 길이
        shift: 시프트 크기 (None이면 랜덤)
    
    Returns:
        np.ndarray: 시프트된 시계열 (T,)
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    if shift is None:
        shift = np.random.randint(0, T)
    
    # Circular shift
    y_shifted = np.roll(y, shift)
    
    return y_shifted


def block_bootstrap(
    y: np.ndarray,
    block_len: int,
    T: int = None
) -> np.ndarray:
    """
    Block bootstrap: 블록 단위로 재표본
    
    Args:
        y: 원본 시계열 (T_orig,)
        block_len: 블록 길이
        T: 생성할 시계열 길이 (None이면 원본 길이)
    
    Returns:
        np.ndarray: 재표본된 시계열 (T,)
    """
    y = np.asarray(y).flatten()
    T_orig = len(y)
    T = T if T is not None else T_orig
    
    # Circular extension for wrapping
    y_extended = np.concatenate([y, y])
    
    result = []
    n_blocks = int(np.ceil(T / block_len))
    
    for _ in range(n_blocks):
        # 블록 시작 위치 선택
        start_idx = np.random.randint(0, T_orig)
        block = y_extended[start_idx:start_idx + block_len]
        result.extend(block)
    
    return np.array(result[:T])


def wmcr_pvalue_calibration(
    y: np.ndarray,
    bands_L: np.ndarray,
    bands_U: np.ndarray,
    p_targets: np.ndarray,
    weights: np.ndarray,
    B: int = 1000,
    block_len: int = 10,
    method: Literal["circular_shift", "block_bootstrap"] = "circular_shift",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    mode: Literal["absolute", "squared", "normalized"] = "absolute",
    random_seed: int = None
) -> dict:
    """
    WMCR 검정의 p-value 계산 (캘리브레이션 검정)
    
    Args:
        y: 실제 관측값 시계열 (T,)
        bands_L: 각 밴드의 하한 (K x T) 또는 (T,) if K=1
        bands_U: 각 밴드의 상한 (K x T) 또는 (T,) if K=1
        p_targets: 각 밴드의 목표 커버리지 (K,)
        weights: 각 밴드의 가중치 (K,)
        B: 부트스트랩 반복 횟수
        block_len: 블록 길이 (시계열 의존성 고려)
        method: 재표본 방법
            - "circular_shift": 순환 시프트 (H0 하에서 y와 밴드 간 의존성 제거)
            - "block_bootstrap": 블록 부트스트랩
        alternative: 대립가설 방향
            - "two-sided": |WMCR - ideal|이 클수록 기각
            - "greater": WMCR이 클수록 기각 (과소 커버 검정)
            - "less": WMCR이 작을수록 기각 (과대 커버 검정)
        mode: WMCR 계산 모드 (compute_wmcr와 동일)
        random_seed: 랜덤 시드
    
    Returns:
        dict: {
            'wmcr_obs': 관측 WMCR,
            'wmcr_null': 귀무분포 WMCR 값들 (B개),
            'pvalue': p-value,
            'alternative': 사용된 대립가설,
            'method': 사용된 방법,
            'B': 반복 횟수,
            'block_len': 블록 길이,
            'coverage_rates': 각 밴드의 관측 캡처율 (K,),
            'target_coverage': 목표 커버리지 (K,)
        }
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    y = np.asarray(y).flatten()
    bands_L = np.asarray(bands_L)
    bands_U = np.asarray(bands_U)
    p_targets = np.asarray(p_targets).flatten()
    weights = np.asarray(weights).flatten()
    
    T = len(y)
    K = len(p_targets)
    
    # 밴드 차원 조정
    if bands_L.ndim == 1:
        bands_L = bands_L.reshape(1, -1)
    if bands_U.ndim == 1:
        bands_U = bands_U.reshape(1, -1)
    
    # 관측 WMCR 계산
    wmcr_obs = compute_wmcr(y, bands_L, bands_U, p_targets, weights, mode=mode)
    
    # 각 밴드의 관측 캡처율 계산 (반환용)
    coverage_rates = np.zeros(K)
    for k in range(K):
        I_tk = (bands_L[k, :] <= y) & (y <= bands_U[k, :])
        coverage_rates[k] = np.mean(I_tk)
    
    # 귀무분포 생성
    wmcr_null = np.zeros(B)
    
    for b in range(B):
        if method == "circular_shift":
            # Circular shift: y를 시프트하여 y와 밴드 간 시간적 의존성 제거
            shift = np.random.randint(0, T)
            y_resampled = circular_block_shift(y, block_len, shift=shift)
        elif method == "block_bootstrap":
            # Block bootstrap: 블록 단위 재표본
            y_resampled = block_bootstrap(y, block_len, T=T)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 재표본된 y로 WMCR 계산
        wmcr_null[b] = compute_wmcr(y_resampled, bands_L, bands_U, p_targets, weights, mode=mode)
    
    # p-value 계산
    if alternative == "two-sided":
        # 이상적인 WMCR = 1.0 (모든 밴드가 목표 커버리지와 정확히 일치)
        ideal_wmcr = 1.0
        deviation_obs = abs(wmcr_obs - ideal_wmcr)
        deviation_null = abs(wmcr_null - ideal_wmcr)
        pvalue = (1 + np.sum(deviation_null >= deviation_obs)) / (B + 1)
    elif alternative == "greater":
        # WMCR이 클수록 기각 (과소 커버 검정)
        pvalue = (1 + np.sum(wmcr_null >= wmcr_obs)) / (B + 1)
    elif alternative == "less":
        # WMCR이 작을수록 기각 (과대 커버 검정)
        pvalue = (1 + np.sum(wmcr_null <= wmcr_obs)) / (B + 1)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    return {
        'wmcr_obs': float(wmcr_obs),
        'wmcr_null': wmcr_null.tolist(),
        'pvalue': float(pvalue),
        'alternative': alternative,
        'method': method,
        'B': B,
        'block_len': block_len,
        'coverage_rates': coverage_rates.tolist(),
        'target_coverage': p_targets.tolist(),
        'mode': mode
    }


# ============================================================================
# 알고리즘 (의사코드)
# ============================================================================
"""
알고리즘: WMCR 캘리브레이션 검정

입력:
  - y: 실제 관측값 시계열 (T,)
  - bands_L, bands_U: 각 밴드의 하한/상한 (K x T)
  - p_targets: 목표 커버리지 (K,)
  - weights: 가중치 (K,)
  - B: 부트스트랩 반복 횟수
  - block_len: 블록 길이
  - method: "circular_shift" 또는 "block_bootstrap"
  - alternative: "two-sided", "greater", "less"

단계 1: 관측 WMCR 계산
  1.1 각 밴드 k=1..K에 대해:
      I_{t,k} = 1{bands_L[k,t] <= y[t] <= bands_U[k,t]} for t=1..T
      C_k = (1/T) * sum_t I_{t,k}
  1.2 각 밴드 k에 대해:
      g_k = 1 - |C_k - p_k|  (또는 다른 정규화 함수)
  1.3 WMCR_obs = sum_k (w_k * g_k) / sum_k w_k

단계 2: 귀무분포 생성
  for b = 1 to B:
    if method == "circular_shift":
      shift = random(0, T-1)
      y_resampled = circular_shift(y, shift)
    else if method == "block_bootstrap":
      y_resampled = block_bootstrap(y, block_len)
    
    WMCR_b = compute_wmcr(y_resampled, bands_L, bands_U, p_targets, weights)
    WMCR_null[b] = WMCR_b

단계 3: p-value 계산
  if alternative == "two-sided":
    ideal = 1.0
    deviation_obs = |WMCR_obs - ideal|
    deviation_null = |WMCR_null - ideal|
    pvalue = (1 + #{deviation_null >= deviation_obs}) / (B+1)
  else if alternative == "greater":
    pvalue = (1 + #{WMCR_null >= WMCR_obs}) / (B+1)
  else if alternative == "less":
    pvalue = (1 + #{WMCR_null <= WMCR_obs}) / (B+1)

출력:
  - wmcr_obs: 관측 WMCR
  - pvalue: p-value
  - wmcr_null: 귀무분포 (B개 값)
  - coverage_rates: 각 밴드의 관측 캡처율
"""


def wmcr_binomial_test(
    T: int,
    p_targets: np.ndarray,
    C_obs: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    WMCR 이항비율 근사 검정
    
    각 밴드에 대해 이항비율 근사로 표준오차를 계산하고,
    허용 오차폭을 기준으로 캘리브레이션을 판정한다.
    
    Args:
        T: 테스트 길이 (시계열 길이)
        p_targets: 각 밴드의 목표 커버리지 (K,)
        C_obs: 각 밴드의 관측 캡처율 (K,)
        alpha: 유의수준 (기본값: 0.05)
        verbose: 상세 출력 여부
    
    Returns:
        dict: {
            'summary_table': 판정 결과 표 (DataFrame),
            'n_acceptable': 허용 가능한 밴드 수,
            'n_total': 전체 밴드 수,
            'all_acceptable': 모든 밴드가 허용 가능한지 여부,
            'problematic_bands': 문제가 있는 밴드 인덱스 리스트,
            'summary_text': 전체 요약 텍스트 (2문장)
        }
    """
    p_targets = np.asarray(p_targets).flatten()
    C_obs = np.asarray(C_obs).flatten()
    
    if len(p_targets) != len(C_obs):
        raise ValueError(f"p_targets와 C_obs의 길이가 일치하지 않음: {len(p_targets)} vs {len(C_obs)}")
    
    K = len(p_targets)
    z_critical = norm.ppf(1 - alpha / 2)  # z_{1-alpha/2} (alpha=0.05일 때 약 1.96)
    
    # 각 밴드에 대해 계산
    results = []
    problematic_bands = []
    
    for k in range(K):
        p_k = p_targets[k]
        C_obs_k = C_obs[k]
        
        # 1) 표준오차: SE_k = sqrt(p_k * (1 - p_k) / T)
        SE_k = np.sqrt(p_k * (1 - p_k) / T)
        
        # 2) 허용 오차폭: tol_k = z_{1-alpha/2} * SE_k
        tol_k = z_critical * SE_k
        
        # 3) 차이 및 절대차이
        diff_k = C_obs_k - p_k
        abs_diff_k = abs(diff_k)
        
        # 4) 판정
        verdict = 'ACCEPTABLE' if abs_diff_k <= tol_k else 'NOT ACCEPTABLE'
        
        if verdict == 'NOT ACCEPTABLE':
            problematic_bands.append(k)
        
        results.append({
            'band': k + 1,
            'p_k': p_k,
            'C_obs_k': C_obs_k,
            'diff_k': diff_k,
            'abs_diff_k': abs_diff_k,
            'SE_k': SE_k,
            'tol_k': tol_k,
            'verdict': verdict
        })
    
    # 결과 표 생성
    summary_table = pd.DataFrame(results)
    
    # 전체 요약
    n_acceptable = np.sum(summary_table['verdict'] == 'ACCEPTABLE')
    n_total = K
    all_acceptable = (n_acceptable == n_total)
    
    # 요약 텍스트 생성 (2문장)
    if len(problematic_bands) == 0:
        summary_text = (
            f"모든 밴드({n_total}개)가 허용 가능한 범위 내에 있어 캘리브레이션이 양호합니다. "
            f"전반적으로 모델의 예측 구간이 목표 커버리지와 일치합니다."
        )
    else:
        band_names = ', '.join([f"밴드 {b+1}" for b in problematic_bands])
        summary_text = (
            f"{band_names}에서 관측 캡처율이 목표 커버리지와 유의한 차이를 보입니다. "
            f"전반적으로 {n_acceptable}/{n_total}개 밴드만 허용 가능하여 캘리브레이션 개선이 필요합니다."
        )
    
    result_dict = {
        'summary_table': summary_table,
        'n_acceptable': int(n_acceptable),
        'n_total': int(n_total),
        'all_acceptable': bool(all_acceptable),
        'problematic_bands': problematic_bands,
        'summary_text': summary_text,
        'alpha': alpha,
        'z_critical': float(z_critical),
        'T': int(T)
    }
    
    # 상세 출력
    if verbose:
        print("\n" + "=" * 80)
        print("WMCR 이항비율 근사 검정 결과")
        print("=" * 80)
        print(f"테스트 길이 T = {T}")
        print(f"유의수준 α = {alpha} (z_{1-alpha/2:.2f} = {z_critical:.4f})")
        print("\n판정 결과 표:")
        print(summary_table.to_string(index=False))
        print("\n" + "-" * 80)
        print("전체 요약:")
        print(summary_text)
        print("=" * 80 + "\n")
    
    return result_dict
