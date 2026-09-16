# 연구 파이프라인 정리 — 한국형 비트코인 현물 ETF 가격 시뮬레이션

> 정리 기준일: 2026-09-07 / 기준 사양: NAV = BTC 현물 + 1거래일 시차보정 (결정 11)
> 표본: 2024-01-12 ~ 2025-05-22, 거래일 342일, 몬테카를로 N = 1,000, seed = 42

이 문서는 **파이프라인 지도**다 — 어떤 코드가 어떤 순서로 무엇을 만드는지를 프레임워크
11단계 구조로 정리했다.

| 문서 | 역할 |
|---|---|
| **이 문서** | 파이프라인 구조, 모듈·산출물 지도, 재현 절차 |
| [DECISIONS_AND_RESULTS.md](DECISIONS_AND_RESULTS.md) | **결과와 결정의 기록.** 수치는 여기가 기준이다 |
| [DISCUSSION.md](DISCUSSION.md) | 보조 진단 2건 (GAP `delta2`, 꼬리 비대칭) |

---

## 0. 핵심 아이디어

한국에는 비트코인 현물 ETF가 없다. 상장된다고 가정하고 그 가격을 세 성분의 곱으로
분해해 각각 별도 확률과정으로 모델링한 뒤, 결합해 가격 경로를 생성한다.

```
ETF_KR(t) = NAV(t) × (1 + GAP(t)) × (1 + KP(t))
```

| 성분 | 뜻 | 모형 | 실측 계열 |
|---|---|---|---|
| **NAV** | 기초자산(BTC) 가치 | ARIMAX(1,0,1)-GARCH(1,1)-t | blockchain.info BTC 현물 (시차보정) |
| **GAP** | ETF 괴리율 | OU + 외생변수 연동 + Student-t | `etf_true/nav_true − 1` (IBIT) |
| **KP** | 김치프리미엄 | Threshold-OU 2레짐 + Student-t | Upbit/Binance |

**검증의 뼈대**: 한국형 ETF는 실재하지 않으므로 3자 결합에는 대조할 실측 계열이 없다.
그러나 앞의 두 성분 `NAV × (1+GAP)`에는 실측 대응물이 있다 — **미국 현물 BTC ETF의
시장가격**이다. 거기까지를 실측으로 검증하고(Step 6-A), 그 결합기에 한국 고유
프리미엄을 얹는다(Step 7).

---

## 1. 파이프라인 11단계

### Phase 1 — Base 시뮬레이션 (Step 1~7)

| Step | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 1 | 데이터 수집 | `data_variables.py`, `data_Modeling.py`, `data_feature.py`, `kp_data.py`, `trend/fetch_trends.py` | `dataset/raw/` |
| 2 | 학습 데이터 생성 | `dataset/make_y_variables.py`, `make_train_data.py`, `make_train_data_ver1.py` | `dataset/train/`, `dataset/train_ver1/` |
| 3 | 모형 선정 | 선행연구 + 데이터 특성 (`analysis/data_characteristics.py`) | `results/data_characteristics/` |
| 4 | 개별 시뮬레이터 검증 | `simulator/main.py` (NAV·GAP·KP 각각) | `results/simulator/validation_results.json` |
| 5 | 시뮬레이터 결합 | `simulator/main.py` | `results/simulator/combined_simulation_results.csv` |
| **6-A** | **미국 ETF 결합 검증 (주검증)** | `simulator/main.py` | `validation_results.json` → `us_etf` |
| 6-B | 3자 결합 검증 (보조) | `simulator/main.py` | `validation_results.json` → `combined` |
| 7 | 한국형 ETF 가격 경로 | `simulator/main.py` (10,000원 앵커) | `results/simulator/korean_etf_price.csv` |

**Step 6-A와 6-B의 지위가 다르다.** 6-A는 관측 가능한 실측 계열(`etf_true`)과 대조하므로
주검증이고, 6-B는 기준이 실측 세 성분의 곱(반사실)이므로 예측 정확도가 아니라 **결합
절차의 정합성**에 대한 보조 검정이다.

### Phase 2 — 시나리오 분석 (Step 8~11)

| Step | 내용 | 코드 | 산출물 |
|---|---|---|---|
| 8 | 레짐 전처리 + 시나리오 설계 | `preprocessing/run_pipeline.py`, `analysis/pairwise_regime_cooccurrence.py`, `analysis/1_scenario_selection.py` | `results/preprocessing/`, `results/scenario_selection/` |
| 9 | 시나리오별 외생변수 생성 | `simulator/scenario_main.py` | `results/scenario_exog_vars/` |
| 10 | 시나리오별 시뮬레이션·검증 | `simulator/scenario_main.py` | `results/scenario_simulator/` |
| 11 | Base 대비 유의성 검정 | `simulator/results_main.py` | `results/scenario_simulator/significance_test/` |

**Step 8은 3년 확장 표본(`dataset/train_ver1/`, 2022-05-02 ~ 2025-05-23, 750일)** 을 쓴다.
레짐 구조는 긴 표본에서 식별하고 시뮬레이션은 342일 표본에서 수행하는 이원 구조이며,
프레임워크가 Phase 1과 Phase 2를 분리하므로 의도된 설계다(논문에 명시 필요).

> **Step 10은 모수를 재추정하지 않는다.** Base 모수를 고정하고 외생변수만 교체한다.
> 프레임워크 그림 표기와 어긋나므로 하나로 맞춰야 한다
> ([DECISIONS_AND_RESULTS.md](DECISIONS_AND_RESULTS.md) 13.1절).

---

## 2. 검증 체계

**검정 3종** (프레임워크가 규정)

| 검정 | 무엇을 보나 |
|---|---|
| PIT-KS | 실제 값이 시뮬 분포 안에서 균등하게 흩어져 있는가 |
| VaR-Kupiec | 5% VaR 초과 횟수가 명목수준과 맞는가 (LR_uc) |
| ES | 초과 시 평균 손실 크기 (모수적 부트스트랩 p-value) |

**검증 대상 5종**

| 단계 | 대상 | 기준 계열 | 성격 |
|---|---|---|---|
| 4 | NAV / GAP / KP | 각 성분의 실측 계열 | 개별 검증 |
| **6-A** | NAV × (1+GAP) | **실측 `etf_true`** | **결합 주검증** |
| 6-B | NAV × (1+GAP) × (1+KP) | 실측 성분곱 대리 계열 | 결합 정합성 (보조) |

현재 사양에서 **5개 전부 통과**한다. 수치는
[DECISIONS_AND_RESULTS.md](DECISIONS_AND_RESULTS.md) 9.2절 참조.

---

## 3. 재현 절차

```bash
# Phase 1 — Base 시뮬레이션 및 검증
python simulator/main.py

# 주요 옵션
python simulator/main.py --nav-source btc        # 시차보정 없는 BTC 현물 (구 사양)
python simulator/main.py --nav-source nav_true   # IBIT 주당 NAV (펀드 비용 포함)
python simulator/main.py --gap-dist normal       # GAP 혁신항을 정규분포로
python simulator/main.py --kp-dist normal        # KP 혁신항을 정규분포로
python simulator/main.py --kp-regimes 3          # KP를 3레짐으로
python simulator/main.py --no-cache              # 캐시를 읽지 않고 강제 재실행

# Phase 2 — 시나리오 시뮬레이션 및 유의성 검정
python simulator/scenario_main.py --scenarios S01,S02,S03,S04,S05,S06,S07,S08,S09
python simulator/results_main.py  --scenarios S01,S02,S03,S04,S05,S06,S07,S08,S09

# 보조 분석 (본 파이프라인과 독립)
python analysis/data_characteristics.py          # Step 3 데이터 특성 근거
python analysis/gap_delta2_diagnostic.py         # GAP delta2 유의성 (약 5분)
python analysis/scenario_distribution_shape.py   # 시나리오별 분포 형태 (약 3분)
python analysis/fanchart.py                      # 발표용 팬차트 2종
```

**소요 시간**: Phase 1 약 2분, Step 9·10 약 2분, **Step 11 약 35분**(100회 반복 × CRN × 두 기준).

`scenario_main.py`와 `results_main.py`는 `--scenarios`를 생략하면 `input()`으로 묻는다.
비대화형(리다이렉트·CI·백그라운드)에서는 EOFError로 죽으므로 ID를 명시해야 한다.

`--nav-source`를 바꾸면 표본 구간(`btc`는 343일, 나머지는 342일이며 시작·종료일도 다르다)과
캐시 키가 함께 바뀌므로 Phase 1을 먼저 다시 돌려야 한다. Phase 2는 `sim_meta.json`의
`nav_source`를 읽어 자동으로 따라간다.

---

## 4. 산출물 지도

| 경로 | 내용 |
|---|---|
| `results/simulator/` | **Phase 1 최종 결과** — `validation_results.json`, 성분별 CSV, `korean_etf_price.csv`, plots, cache |
| `results/simulator/cache/` | `sim_arrays.npz`(MC 경로 4종), `sim_meta.json`(모수·사양). **Phase 2가 이걸 읽는다** |
| `results/scenario_simulator/` | 시나리오별 결과(S01~S09), `comparison_summary.csv`, `korean_etf_risk_metrics.csv` |
| `results/scenario_simulator/significance_test/` | **Step 11 결과.** `effect_size_comparison.csv`(요약), `*_etf.csv` / `*_kr.csv`(기준별 상세) |
| `results/figures/` | 발표용 팬차트 (`fanchart_us`, `fanchart_kr` — png/svg) |
| `results/data_characteristics/` | Step 3 데이터 특성 검정 |
| `results/discussion/` | DISCUSSION.md 진단 2건의 원자료 |
| `results/preprocessing/` | HAR-VKOSPI, Gaussian HMM, Bai-Perron 플롯 |
| `results/scenario_selection/` | 시나리오 후보·최종 (`final_scenarios_latest.csv`) |
| `results/scenarios/`, `results/scenario_exog_vars/` | 시나리오별 레짐 플롯 / 외생변수 시계열 |
| `results/compare_results/`, `results/gap_results/` | **모형 후보 비교 (코드는 삭제됨, 산출물만 잔존)** |
| `results/_archive_pre_decision11/` | 결정 11 이전 산출물 보관 |

### 4.1 유의성 검정 결과 읽는 법

**반드시 `_kr`(한국 요인) 기준을 주 결과로 쓸 것.** `_etf` 기준은 NAV의 누적 변동이
분산을 지배해 시나리오 효과가 희석되고, 위기 시나리오에서 VaR 부호가 뒤집힌다.
`_etf`는 "왜 `_kr` 기준을 채택했는지" 보이는 대조군으로만 쓴다
([DECISIONS_AND_RESULTS.md](DECISIONS_AND_RESULTS.md) 10.4절).

---

## 5. 현재 사양 요약

| 항목 | 값 |
|---|---|
| 표본 | 2024-01-12 ~ 2025-05-22, 342 거래일 |
| NAV | BTC 현물 로그수익률 + 1거래일 시차보정, S0 = 24.94달러 |
| NAV 모형 | ARIMAX(1,0,1)-GARCH(1,1)-t, ν = 4.300 |
| GAP 모형 | OU + Student-t, κ = 0.9345, ν = 5.074 |
| KP 모형 | Threshold-OU 2레짐 + Student-t, τ = 0.043075, ν = 12.382 |
| 몬테카를로 | N = 1,000, seed = 42 |
| 시나리오 | S01~S09 (5개 외생변수 레짐 조합) |
| Step 11 | 100회 반복 × CRN, 두 기준(etf / kr) |

**Base Volatility 절대값** (일별 로그수익률 표준편차 — 효과크기가 상대값이라 기준값이 필요)

| 기준 | 일별 | 연율화 |
|---|---|---|
| etf (전체) | 0.036834 | 0.5847 |
| kr (한국 요인) | 0.012585 | 0.1998 |

---

## 6. 남은 과제

우선순위 순이다. 상세는 [DECISIONS_AND_RESULTS.md](DECISIONS_AND_RESULTS.md) 13장 참조.

1. **Step 10의 "모수 재추정" 표기 불일치** — 그림·주석·실제 동작을 하나로 맞춰야 한다
2. **GAP `delta2` 처리** — 자유 추정 유지 + 한계 서술 / `delta2 = 0` 제약 중 택일
3. **NAV 시차보정의 근본 해결** — 16:00 ET 기준 BTC 레퍼런스 레이트 수집
4. `etf/` 가상환경과 `__pycache__` 추적 제외
5. `results/compare_results/`, `results/gap_results/`는 코드가 삭제돼 재현 불가 — 인용 시 명시 필요
