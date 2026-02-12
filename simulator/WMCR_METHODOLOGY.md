# Weighted Multi-band Capture Rate (WMCR) 검정 통계량 방법론

## 1. 방법론 설명 (논문용)

Weighted Multi-band Capture Rate (WMCR)는 모델의 예측 구간 캘리브레이션을 평가하는 검정 통계량이다. 
K개의 예측 구간 밴드 [L_{t,k}, U_{t,k}] (k=1,...,K)가 각 시점 t=1,...,T에 주어지고,
각 밴드 k는 목표 커버리지 p_k와 중요도 가중치 w_k를 가진다.

밴드 k의 관측 캡처율은 다음과 같이 정의된다:

$$C_k = \frac{1}{T} \sum_{t=1}^T I_{t,k}$$

여기서 $I_{t,k} = \mathbf{1}\{L_{t,k} \leq y_t \leq U_{t,k}\}$는 실제 관측값 $y_t$가 밴드 내에 있는지 여부를 나타내는 지시함수이다.

WMCR 통계량은 다음과 같이 정의된다:

$$\text{WMCR} = \frac{\sum_{k=1}^K w_k \cdot g(C_k, p_k)}{\sum_{k=1}^K w_k}$$

여기서 $g(C_k, p_k) = 1 - |C_k - p_k|$는 캡처율이 목표 커버리지에 가까울수록 큰 값을 반환하는 정규화 함수이다.
WMCR은 0과 1 사이의 값을 가지며, 1에 가까울수록 모든 밴드에서 목표 커버리지에 근접함을 의미한다.

귀무가설 $H_0$는 "모든 밴드에서 $C_k = p_k$ (캘리브레이션)"이고,
대립가설 $H_1$은 "적어도 하나의 밴드에서 체계적 과대/과소 커버"이다.

시계열 의존성을 고려하여 p-value를 계산하기 위해, circular block shift 방법을 사용한다.
이 방법은 관측 시계열 $y_t$를 블록 단위로 순환 시프트하여 $y_t$와 밴드 $[L_{t,k}, U_{t,k}]$ 간의
시간적 의존성을 깨뜨림으로써 $H_0$ 하에서의 귀무분포를 생성한다.
B회의 재표본을 통해 $\text{WMCR}_{\text{null}}$ 분포를 구성하고,
p-value = $(1 + \#\{\text{WMCR}_{\text{null}} \geq \text{WMCR}_{\text{obs}}\}) / (B+1)$로 계산한다.

## 2. 알고리즘 (의사코드)

```
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
```

## 3. 파이썬 함수 스펙

### 3.1 `compute_wmcr`

```python
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
```

### 3.2 `wmcr_pvalue_calibration`

```python
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
```

## 4. 사용 예제

```python
import numpy as np
from simulator.wmcr_test import compute_wmcr, wmcr_pvalue_calibration

# 데이터 준비
T = 100
K = 3
y = np.random.randn(T)  # 실제 관측값
bands_L = ...  # (K x T) 하한 배열
bands_U = ...  # (K x T) 상한 배열
p_targets = np.array([0.50, 0.80, 0.95])
weights = np.array([3, 2, 1])

# WMCR 계산
wmcr = compute_wmcr(y, bands_L, bands_U, p_targets, weights)

# WMCR 검정 (p-value)
result = wmcr_pvalue_calibration(
    y=y,
    bands_L=bands_L,
    bands_U=bands_U,
    p_targets=p_targets,
    weights=weights,
    B=1000,
    block_len=10,
    method="circular_shift",
    alternative="two-sided"
)

print(f"WMCR: {result['wmcr_obs']:.4f}")
print(f"p-value: {result['pvalue']:.4f}")
```

## 5. 해석

- **WMCR = 1.0**: 모든 밴드에서 관측 캡처율이 목표 커버리지와 정확히 일치 (완벽한 캘리브레이션)
- **WMCR < 1.0**: 일부 밴드에서 목표 커버리지와 차이 존재
- **p-value < 0.05**: H0 기각 → 모델이 목표 커버리지와 일치하지 않음 (캘리브레이션 실패)
- **p-value >= 0.05**: H0 채택 → 모델이 목표 커버리지와 일치함 (캘리브레이션 성공)
