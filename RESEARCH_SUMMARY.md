# 연구 진행 정리 — Bitcoin ETF Price Estimation under Korean Financial Market

> A Scenario-based Simulation
> 정리 기준일: 2026-08-28 / 기준 커밋: `8b55590 (20260714_1)`

---

## 0. 핵심 아이디어

한국형 비트코인 ETF 가격을 **3개 컴포넌트의 곱**으로 분해하여 각각 별도 확률과정으로 모델링하고,
결합한 뒤 레짐 기반 시나리오별 리스크를 평가한다.

```
ETF_KR(t) = NAV(t) x (1 + GAP(t)) x (1 + KP(t))      <- 초기값 10,000원 앵커
              |          |               |
              |          |               +-- 김치프리미엄  : Threshold-OU (3레짐)
              |          +------------------ ETF 괴리율    : OU + 외생변수 연동
              +----------------------------- 기초자산 NAV  : ARIMAX-GARCH-t
```

---

## 1. 전체 파이프라인 흐름

```
[1] 데이터 수집
    data_variables.py / data_Modeling.py / data_feature.py / kp_data.py / trend/fetch_trends.py
        |
        v  dataset/raw/*.csv
[2] 학습데이터 생성
    dataset/make_train_data.py       -> dataset/train/      (2024-01-12 ~ 2025-05-23, 약 343일)
    dataset/make_train_data_ver1.py  -> dataset/train_ver1/ (2022-05-01 ~ 2025-05-23, 3년 확장)
        |
        +--> [3] 모델 후보 비교 (compare/, compare_gap/, compare_kp/, btc_etf_valid/)
        |         -> results/compare_results, results/gap_results
        |
        +--> [4] 본 모델 Base 시뮬레이션 (simulator/main.py)          [train/ 사용]
        |         -> results/simulator/
        |
        +--> [5] 레짐 전처리 (preprocessing/run_pipeline.py)          [train_ver1/ 사용]
                  HAR-VKOSPI -> Gaussian HMM (5개 변수)
                      |
                      v
             [6] 시나리오 선정 (analysis/)
                  pairwise_regime_cooccurrence.py -> 1_scenario_selection.py
                      -> results/scenario_selection/final_scenarios_latest.csv (S01~S09)
                      |
                      v
             [7] 시나리오 시뮬레이션 (simulator/scenario_main.py)
                      -> results/scenario_simulator/
                      |
                      v
             [8] Base 대비 유의성 검정 (simulator/results_main.py)
                      -> results/scenario_simulator/significance_test/
```

---

## 2. 단계별 상세

### 2.1 데이터 수집

| 파일 | 역할 | 출처 |
|---|---|---|
| `data_variables.py` | 온체인 지표 + Google Trends + 환율/금리 수집 | blockchain.info, yfinance, holidays |
| `data_Modeling.py` | 위와 유사 + pyupbit 국내 거래량 | blockchain.info, yfinance, pyupbit |
| `data_feature.py` | 수집(1~11) + 보조지표 + 마스터 병합, CLI 형태 | CoinGecko, Upbit, Binance, FRED, BOK |
| `kp_data.py` | 김치프리미엄 계산: `BTC_KRW / (BTC_USD x USD_KRW) - 1` | 로컬 CSV |
| `trend/fetch_trends.py` | Google Trends SVI (글로벌 `bitcoin_all_`, 한국 `bitcoin_kr_`) | pytrends |
| `dataset/make_y_variables.py` | `y_variables.csv` 생성 (`Log Return` 재현 경로) | blockchain.info |

수집 기간(주 분석): **2024-01-12 ~ 2025-05-23**

**`Log Return` 정의** — 거래일 그리드(미국 영업일 − 연방공휴일, 343일)로 **먼저 필터링한 뒤 로그차분**한다.
BTC는 24/7 거래되므로 전체 달력에서 먼저 로그차분하면 `shift(1)`이 1 거래일이 아니라 1 달력일 차분이 되고,
이후 주말 행을 제거할 때 금→월 이동이 통째로 사라진다. 필터를 먼저 적용하면 금→월이 하나의 관측치로 보존된다.
`etf_premium`(= `etf_true/nav_true − 1`)과 `Kimchi Premium`은 수준 비율이라 차분 순서의 영향을 받지 않는다.
`btc_volatility`(→ `Global_RV`)는 전체 달력에서 RV를 **집계한 뒤** 거래일로 샘플링하므로 동일 이슈가 없다.

### 2.2 학습 데이터

| 디렉터리 | 기간 | 파일 | 사용처 |
|---|---|---|---|
| `dataset/train/` | 2024-01-12 ~ 2025-05-23 (343일) | `nav_train.csv`, `gap_train_main.csv`, `kp_train_main.csv` | `simulator/` (Base 시뮬레이션) |
| `dataset/train_ver1/` | 2022-05-01 ~ 2025-05-23 (3년) | `nav_train.csv`, `gap_train.csv`, `kp_train.csv` | `preprocessing/`, `analysis/` (HMM·시나리오) |

주요 변수 매핑:

| 논리명 | 소스 컬럼 | 설명 |
|---|---|---|
| `global_btc_svi` | `gap_train["value"]` | Google Trends 글로벌 (0~100) |
| `domestic_btc_svi` | `kp_train["bitcoin_kr"]` | Google Trends 한국 (0~100) |
| `btc_volume_btc` | `kp_train["volume_btc"]` | 국내 BTC 일별 거래량 |
| `VKOSPI` | `kp_train["KOSPI_Volatility"]` | 일별 VKOSPI (ver1은 ^KS11 30일 RV로 대체) |
| `Global_RV` | `gap_train["btc_volatility"]` | BTC 30일 롤링 실현변동성 |

### 2.3 모델 후보 비교 (벤치마킹)

각 컴포넌트에 어떤 확률과정이 적합한지 사전 검증.

**NAV** — `compare/compare_main.py` (N=5000)

| 모델 | PIT-KS p | Kupiec p | is_valid |
|---|---|---|---|
| GBM | 0.569 | 0.057 | O |
| GARCH | 0.151 | **0.005** | **X** |
| Heston | 0.773 | 0.106 | O |
| Poisson-Gaussian | 0.414 | 0.483 | O |
| Merton-JD | 0.295 | 0.783 | O |
| ARIMA-GARCH | (구현 존재) | | |

**GAP** — `compare_gap/gap_main.py`

| 모델 | PIT-KS p | Kupiec p | is_valid |
|---|---|---|---|
| OU | 0.190 | 0.946 | O |
| Heston-SV | 0.359 | 0.946 | O |
| GARCH | **1.13e-21** | 1.000 (초과 0건) | **X** |

**KP** — `compare_kp/kp_main.py` : OU, Heston-SV를 gap과 동일 절차로 비교

**ETF 조합 검정** — `btc_etf_valid/etf_valid_main.py`
NAV{Heston, ARIMA-GARCH} x GAP{Heston-SV, OU} 4개 조합으로 `ETF = NAV x (1+gap)` 구성 후 `etf_true`와 대조.

**검증 지표 체계** (`compare/metrics.py`, 889줄)
- 통계 검정: PIT-KS, VaR-Kupiec(LR_uc), ES backtest
- 경로/분포 지표: DTW, PMC, RVR, VVS, VPR, VJC, TWAD, KS, 분포 적률, VaR/ES proximity
- **예측구간 커버리지 (PICP / NMPIW)** — `interval_coverage()`, `coverage_metrics()`
  - 각 시점 t에서 MC 경로의 분위수로 명목 (1-a) 예측구간을 만들고, 실제값이 그 안에 들어간 시점의 비율을 센다
  - `PICP_lv` = 경험적 커버리지(명목수준에 가까울수록 좋음), `CovErr` = mean |경험적 - 명목| (낮을수록 좋음)
  - 커버리지만 보면 구간을 넓게 잡을수록 유리하므로, 구간폭을 실제 변동범위로 정규화한
    `NMPIW95`를 함께 본다(낮을수록 좁고 예리한 구간). 명목수준 3종(50/80/95%) 사용

> 이전에는 자체 고안 지표 WMCR(Weighted Multi-band Capture Rate)을 썼으나 제거했다.
> WMCR의 밴드는 `MC 경로 min/max의 중점 +- p%`로 정의되는데, min/max는 표본크기 N에 따라
> 발산하는 통계량이라 밴드폭이 모델 분포가 아니라 N에 끌려간다. 예측구간 커버리지는
> 분위수 기반이라 이 문제가 없고, 예측문헌의 표준 관행이라 별도 방법론 정당화가 필요 없다.

### 2.4 본 모델 (Base) — `simulator/main.py`

| 컴포넌트 | 모형 | 외생변수 |
|---|---|---|
| NAV | ARIMAX(1,0,1)-GARCH(1,1)-t | Hash Rate, Unique Addresses |
| GAP | OU, `mu_t = mu_0 + gamma1*SI`, `sigma_t = sigma_0 * exp(delta1*RV)` | Search Interest, BTC RV |
| KP | Threshold-OU 3레짐 (abs(KP)<=tau / KP>tau / KP<-tau), tau 최적 탐색 | volume_btc, KOSPI_Volatility, bitcoin_kr |

몬테카를로 N=1000, seed=42, MD5 기반 결과 캐싱(`results/simulator/cache/`).

**Base 검증 결과** (`results/simulator/validation_results.json`, T=343)

| 대상 | PIT-KS p | Kupiec p | ES tail_error | is_valid | DTW Price | PMC |
|---|---|---|---|---|---|---|
| NAV | 0.979 | 0.652 | +0.00334 | O | 0.062 | 0.472 |
| GAP | 0.263 | 0.349 | -0.00052 | O | 0.136 | 0.460 |
| KP | 0.904 | 0.243 | -0.00284 | O | 0.160 | 0.509 |
| Combined | 0.354 | 0.652 | -0.00144 | O | 0.062 | 0.467 |

**예측구간 커버리지** (같은 실행, 명목 50/80/95%)

| 대상 | PICP50 | PICP80 | PICP95 | CovErr | NMPIW95 | PICP95(Vol) | CovErr(Vol) |
|---|---|---|---|---|---|---|---|
| NAV | 0.741 | 0.921 | 0.994 | **0.135** | **1.860** | 0.974 | 0.035 |
| GAP | 0.580 | 0.825 | 0.942 | 0.038 | 0.526 | 0.892 | 0.126 |
| KP | 0.467 | 0.811 | 0.945 | **0.017** | 0.642 | 0.868 | 0.074 |
| Combined | 0.618 | 0.907 | 0.997 | 0.091 | **1.767** | 0.869 | 0.093 |

KP(0.017)와 GAP(0.038)은 잘 보정돼 있다. **NAV는 과대분산이다** — 명목 50% 구간이 실제로 74%를
덮고, 95% 구간폭이 실제 시계열 변동범위의 1.86배다. Combined(NMPIW95 = 1.77)가 같은 패턴을
보이는 것은 NAV에서 상속된 결과이며, 4.4-4의 `sim_std` 26배 문제와 같은 원인으로 보인다.

`is_valid`는 PIT-KS와 Kupiec의 p-value만으로 판정한다. ES는 p-value 없는 진단 지표(tail_error)라 판정에 포함하지 않는다.
NAV 적합 모수: mu=0.002538, phi1=-0.494257, theta1=0.427565, omega=3.969064, alpha1=0.070138, beta1=0.610203, nu=4.178673.
Combined 검정은 양측이 대수적으로 GAP과 같아(`nav x (1+gap) / nav - 1 = gap`) NAV 사양 변경에 반응하지 않는다 — 해석 시 유의.

산출물: `nav/gap/kp/combined_simulation_results.csv`, `korean_etf_price.csv`, `plots/{nav,gap,kp,combined}/`

### 2.5 레짐 전처리 — `preprocessing/`

| 모듈 | 내용 |
|---|---|
| `har_vkospi.py` | HAR(d=1, w=5, m=22)로 VKOSPI 적합 -> 평균회귀 사이클 제거한 **표준화 잔차** 추출 |
| `gaussian_hmm.py` | hmmlearn GaussianHMM. K 결정 = BIC(K=2 vs 3) + **Bootstrap LRT B=1000** + 레짐 점유율 >= 10% |
| `asvi_transformer.py` | ASVI = `(X_t - mean(X[t-k:t-1])) / std(X[t-k:t-1])`, k=4주, look-ahead 없음 |
| `scenario_generator.py` | 레짐 조건부 외생변수 시계열 생성 + MC |
| `run_pipeline.py` | Step1 로드 -> Step2 HAR -> Step3 HMM(주별 SVI/Volume) -> Step4 HMM(일별 변동성), pickle 캐시 |

로그 변환: `Global_RV` -> log / SVI·Volume -> log1p / `VKOSPI_resid` -> 없음

**레짐 변수 5종**

| 변수 | 표시명 | 상태 | 주기 |
|---|---|---|---|
| `Global_RV_regime` | Bitcoin_RV | Low / Mid / High | 일별 |
| `VKOSPI_resid_regime` | VKOSPI | Normal / Extreme | 일별 |
| `btc_volume_btc_regime` | KR_Volume | Low / High | 주별 -> ffill |
| `domestic_btc_svi_regime` | KR_SVI | Low / High | 주별 -> ffill |
| `global_btc_svi_regime` | Global_SVI | Low / Mid / High | 주별 -> ffill |

### 2.6 시나리오 선정 — `analysis/`

1. `pairwise_regime_cooccurrence.py` — 5변수 레짐 쌍별 공통빈도 / lift 분석 -> `results/pairwise_regime_cooccurrence_results.xlsx`
2. `1_scenario_selection.py` — MODE A(HMM 분류 테이블 생성) / MODE B(테이블 로드 후 선택). diversity·lift 임계값으로 최종 시나리오 확정.

**최종 시나리오** (`results/scenario_selection/final_scenarios_latest.csv`)

| ID | Bitcoin_RV | VKOSPI | KR_Volume | KR_SVI | Global_SVI | 공동빈도 | lift 평균 |
|---|---|---|---|---|---|---|---|
| S01 | Low | Normal | Low | Low | Mid | 67 | 1.131 |
| S02 | Mid | Normal | High | High | High | 45 | 1.095 |
| S03 | Low | Normal | Low | Low | Low | 23 | 1.169 |
| S04 | Mid | Normal | Low | Low | Low | 22 | 1.140 |
| **S05** | **High** | **Extreme** | **High** | **High** | **High** | 21 | **1.260** |
| S06 | Mid | Normal | High | High | Mid | 17 | 0.942 |
| S07 | Mid | Normal | Low | Low | Mid | 11 | 1.042 |
| S08 | Mid | Normal | High | High | Low | **0** | - |
| S09 | High | Extreme | High | High | Mid | **0** | - |

S05 = 위기 시나리오(고변동성 + KOSPI 극단 + 관심도/거래량 급증). S08·S09는 이론적으로만 정의되고 관측 0건.

### 2.7 시나리오 시뮬레이션 — `simulator/scenario_main.py`

방법: **실제 GAP/KP 시계열은 유지**하고, 외생변수만 시나리오 레짐에서 생성한 시계열로 교체 -> 모수 재추정 -> MC.

외생변수 매핑:
- GAP 모델: `global_btc_svi` -> Search Interest, `Global_RV` -> VIX Volatility
- KP 모델: `btc_volume_btc` -> volume_btc, `VKOSPI_resid` -> KOSPI_Volatility, `bitcoin_kr`은 실제 데이터 유지

커밋 이력상 **두 갈래**가 존재:
- `f14c00a 20260626_para_reestimate` — 시나리오별 모수 재추정
- `df14167 20260626_fixed_parameter_ver` — 모수 고정, 외생변수만 교체

**결과** (`results/scenario_simulator/korean_etf_risk_metrics.csv`)

| Scenario | VaR_95 | CVaR_95 | VaR_99 | CVaR_99 | Max_DD_mean | Volatility | Skew | Kurt | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | 5,951 | 4,679 | 3,794 | 3,508 | -0.41 | 571.7 | 4.00 | 31.80 | 16,612 | 52,261 |
| S05 | 5,936 | 4,693 | 3,797 | 3,524 | -0.41 | 582.4 | 4.19 | 35.47 | 16,674 | 52,361 |

모수 비교 (`comparison_summary.csv`):

| | gap_kappa | gap_mu | gap_sigma0 | gap_delta1(SI) | gap_delta2(VIX) |
|---|---|---|---|---|---|
| Base | 0.9149 | 0.000475 | 0.004118 | 0.1136 | -0.0729 |
| S05 | 0.9375 | 0.000459 | 0.004157 | 0.0279 | 0.0377 |

`comparison_summary.csv`의 WMCR 컬럼(`gap_wmcr_pass`/`kp_wmcr_pass`)은 제거됐고
`gap_picp95`/`gap_cov_err`/`kp_picp95`/`kp_cov_err`로 대체됐다. 위 표의 숫자는
WMCR 제거 이전 실행 기준이므로, 시나리오 시뮬레이션 재실행 후 갱신이 필요하다.

### 2.8 유의성 검정 — `simulator/results_main.py` (최신 작업)

시나리오 지표 차이가 MC 노이즈인지, Base 대비 통계적으로 유의한 차이인지 검정.

절차:
1. Base/각 시나리오를 **M=100회 독립 반복** x N=1000경로. 반복 m마다 **공통 난수(CRN)** 사용해 base_m과 scenario_m을 짝지음.
2. GAP/KP 모수는 Base·시나리오 공통 고정, NAV 경로는 캐시 재사용 -> 차이는 오직 외생변수에서만 발생
3. 반복별 `diff_m = Scenario_m - Base_m`
4. Shapiro-Wilk 정규성 검정 -> 만족 시 대응표본 t-검정, 위반 시 Wilcoxon 부호순위(정규근사)
5. p < 0.05 -> 유의. 다중비교 보정 미적용 (Rothman 1990: 사전 설계된 시나리오이므로 만능귀무가설 부적절)

**판정 기준** — M=100 대응표본에 CRN을 적용하면 SE가 매우 작아져 0.5% 수준의 차이도 p<0.05가 된다
(실제로 수정 전 계열에서는 8지표 x 9시나리오 = 72개 비교가 전부 유의했다). p-value만으로는 변별이 되지 않으므로
**유의성과 효과크기를 함께** 본다. 임계값은 지나치게 좁히지 않도록 완만하게 잡았다.

> `priority` = p < 0.05 **그리고** |효과크기| >= 0.25%,  그 외 `noise`

효과크기 분포에서 CVaR_99(0.50%)와 VaR_95(0.15%) 사이에 자연스러운 간격이 있어 그 사이에 임계값을 두었다.
0.25%는 8개 지표 중 6개를 `priority`로 남기는 관대한 선이다. 민감도: 0.18~0.25% 구간에서는 아래 표가 그대로이고,
0.45%까지 올려도 `priority` 지표 구성(6개)은 바뀌지 않는다(일부 시나리오에서 Max_DD_mean·CVaR_99가 빠질 뿐).

**결론** (`significance_test/significance_matrix_flag.csv` + `effect_size_matrix_pct.csv`)

| 지표 | 효과크기(중앙) | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|---|---|
| Volatility | 2.95% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| p50 | 0.92% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| VaR_99 | 0.78% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| Max_DD_mean | 0.70% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| CVaR_95 | 0.53% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| CVaR_99 | 0.50% | priority | priority | priority | priority | priority | priority | priority | priority | priority |
| VaR_95 | 0.15% | noise | noise | noise | noise | noise | noise | noise | noise | noise |
| p95 | 0.07% | noise | noise | noise | noise | noise | noise | noise | noise | noise |

-> **시나리오 간 차이는 변동성에서 가장 뚜렷하고**(2.95%, 다른 지표의 3배 이상), 중앙값·꼬리 평균(CVaR)·
최대낙폭·VaR_99에서도 일관되게 나타난다. 반면 **분포 양끝단(VaR_95, p95)에서는 실질적 차이가 없다.**
9개 시나리오 모두 같은 패턴이라 지표별 판정이 시나리오에 따라 흔들리지 않는다.

`p5`는 산출 정의상 `VaR_95`와 항상 같은 값이라 지표 목록에서 제외했다(8개 지표).

**참고** — 이 표는 `Log Return` 수정(2.1 참조) 이후의 결과다. 수정 전에는 72개 비교가 전부 유의해
지표 간 변별이 되지 않았고, 수정 후 VaR_95(0.49%→0.15%)와 p95(0.54%→0.07%)의 효과크기가 줄면서
분포 양끝단이 분리되었다. 수정 전 결과는 `significance_test_preW/`에 보존되어 있다.

---

## 3. 산출물 지도

| 경로 | 내용 |
|---|---|
| `results/preprocessing/` | HAR 플롯, HMM 플롯(hmm_svi/hmm_vol), bai_perron*, ASVI 변환, hmm_cache.pkl |
| `results/regime_classified_table.csv` | 일별 레짐 분류 테이블 (MODE A 산출) |
| `results/pairwise_regime_cooccurrence_results.xlsx` | 레짐 쌍별 공통빈도·lift |
| `results/scenario_selection/` | 시나리오 후보/최종 (xlsx 10개 + final_scenarios_latest.csv) |
| `results/scenarios/` | 시나리오별 히스토그램/레짐 시계열 플롯 (P01~P09) |
| `results/scenario_exog_vars/` | 시나리오별 외생변수 시계열 |
| `results/compare_results/`, `results/gap_results/` | 모델 후보 비교 결과 |
| `results/simulator/` | **Base 시뮬레이션 최종 결과** + plots + cache |
| `results/simulator/scenario_regime/` | gbm_regime_simulator 산출 (아래 이슈 참조) |
| `results/scenario_simulator/` | 시나리오 시뮬레이션 (Base, S05) + significance_test |

### 3.1 다른 PC에서 이어서 작업할 때 — 착수 전 확인

`.gitignore:212`가 `results/` 전체를 제외하므로, **clone만으로는 아래 4개 파일이 없다.**
이들은 산출물이 아니라 다음 단계의 **입력**이라, 없으면 시뮬레이터가 바로 실패한다.
작업 시작 전 존재 여부를 먼저 확인할 것.

```bash
FILES="results/scenario_selection/final_scenarios_latest.csv
results/cache/regime_df_cache.csv
results/cache/scenario_hmm_arrays.npz
results/simulator/cache/sim_arrays.npz"

echo "$FILES" | while read -r f; do
  [ -e "$f" ] && echo "OK   $f" || echo "없음 $f"
done
```

| 파일 | 역할 | 없을 때 |
|---|---|---|
| `results/scenario_selection/final_scenarios_latest.csv` | S01~S09 시나리오 정의 | `scenario_main.py` / `results_main.py` 실행 불가 |
| `results/cache/regime_df_cache.csv` | HMM 레짐 분류 캐시 | `results_main.py` FileNotFoundError |
| `results/cache/scenario_hmm_arrays.npz` | HMM 모수 캐시 | HMM 재적합(`n_init=10, B=1000`) 발생 |
| `results/simulator/cache/sim_arrays.npz` | Base MC 배열 (NAV/GAP/KP, 1000x343) | NAV·GAP·KP 전체 재시뮬레이션 |

**전부 있으면** 그대로 이어서 작업하면 된다. `simulator/main.py`는 데이터 파일 MD5 기반으로
캐시 유효성을 자체 판정하므로(`_sim_cache_key`), 입력이 바뀌었으면 알아서 재계산한다.

**하나라도 없으면** 아래 순서로 재생성한다. 이 경로는 HMM을 다시 적합하므로
S01~S09 레짐 조합이 달라질 수 있다 — 달라지면 2.6~2.8 결과를 다시 확인할 것.

```bash
python preprocessing/run_pipeline.py        # 레짐 분류 + HMM 캐시
python analysis/1_scenario_selection.py     # final_scenarios_latest.csv
python simulator/main.py                    # Base MC + sim_arrays.npz (~5분)
echo all | python simulator/scenario_main.py
echo all | python simulator/results_main.py # M=100 x N=1000, 약 25분
```

`dataset/` 이하는 전부 git 추적 대상이므로 별도 조치가 필요 없다.
`data/`(원천 CSV: `IBIT Premium.csv`, `USD_KRW.csv`, `BTC_KRW.csv`)는 저장소에 없지만,
현재 파이프라인은 이 디렉터리를 쓰지 않는다 — `data_Modeling.py` / `data_variables.py` /
`kp_data.py` 를 직접 재실행할 때만 필요하다.

---

## 4. 정리 후보 — 검토 요청 항목

### 4.1 깨진 임포트 (현재 실행 불가)

| 파일 | 존재하지 않는 대상 |
|---|---|
| `compare/compare_main.py` | `simulator.jump_detector`, `simulator.data_loader.load_nav_data` |
| `compare/poisson_gaussian_simulator.py` | `simulator.nav_simulator`, `simulator.continuous_component`, `simulator.jump_detector` |
| `simulator/gbm_regime_simulator.py` | `analysis.scenario_selection` (실제 파일명 `1_scenario_selection.py` — 숫자 접두사라 import 불가) |

> 해소됨: `simulator/__init__.py`의 `__all__` 잔존 항목 3개는 WMCR 제거 작업에서 함께 삭제했다.

**결정 필요**: 이 파일들을 (a) 고쳐서 살릴지 (b) 삭제할지

### 4.2 최종 파이프라인에서 벗어난 것으로 보이는 코드

| 대상 | 줄수 | 상태 |
|---|---|---|
| `simulation/regime_simulation.py` | 1,016 | 어디서도 import 안 됨. `simulator/scenario_main.py`로 대체된 초기 버전으로 추정. 폰트도 `NanumGothic`(타 파일은 `Malgun Gothic`) |
| `simulator/gbm_regime_simulator.py` | 495 | 깨진 import + 결과 이상(4.4 참조) |
| `data_Modeling.py` / `data_variables.py` / `data_feature.py` | 206/251/395 | blockchain.info 수집 로직 3중 중복 |
| `settings.py` 일부 | - | `ARIMAX_CONFIG`, `GARCHX_CONFIG`, `MULTIPLE_TESTING`, `EDA_*`, `VARIABLES_DIR` 설정이 현행 코드에서 사실상 미사용 (`compare_main.py`가 `PROJECT_ROOT`만 참조) |
| `eda/`, `plot/`, `correlation/` | - | 초기 탐색 단계 산출물. 논문 포함 여부 확인 필요 |
| `etf/` | - | 가상환경 디렉터리가 저장소에 포함됨 |
| `__pycache__/`, `*/__pycache__/` | - | 커밋된 캐시 |
| `trend/bitcoin_all_ (1)~(5).csv` 등 | - | 병합 전 원본 12개. `merged_*.csv`만 있으면 되는지 확인 필요 |
| `results/scenario_selection/*_2026*.xlsx` | 10개 | 타임스탬프 중간 산출물 |

**결정 필요**: 각 항목 삭제 / 보존(논문 부록) / 아카이브 분리

### 4.3 데이터셋 이원화

- `simulator/` (Base·시나리오 시뮬레이션) -> `dataset/train/` (343일)
- `preprocessing/`·`analysis/` (HMM·시나리오 선정) -> `dataset/train_ver1/` (3년)

즉 **레짐을 3년 데이터로 추정하고, 시뮬레이션은 343일 데이터로 수행**하고 있음.
의도된 설계라면 논문에 명시가 필요하고, 아니라면 한쪽으로 통일 필요.

**결정 필요**: 최종본이 `train`인지 `train_ver1`인지

### 4.4 결과 자체의 미완 / 이상

1. **시나리오 시뮬레이션 미완** — `results/scenario_simulator/`에 **Base와 S05만** 존재. S01~S04, S06~S09 미실행.
   (단, `significance_test/`는 S01~S09 전체 결과가 있음 — 별도 실행 경로)
2. **`gbm_regime_simulator` 결과 이상** — `scenario_regime_risk_metrics.csv`에서 P01, P02, P06, P07, P08, P09 6개 시나리오의 `sigma_GAP=0.0903`, `sigma_KP=0.2067`, 모든 리스크 지표가 **완전히 동일**. 레짐 필터가 작동하지 않은 것으로 보임.
3. ~~**GAP WMCR 미통과**~~ — **해소.** 지표 결함이었다. 예측구간 커버리지로 재측정하니 GAP의
   `CovErr = 0.038`로 4개 컴포넌트 중 두 번째로 잘 보정돼 있다. WMCR이 낮게 나온 것은
   밴드를 min/max 중점 기준으로 잡는 정의 탓이지 GAP 모델의 문제가 아니었다(2.3 참조).
4. **NAV·Combined 과대분산** — Combined `mean_proximity ≈ -381.8`, `sim_std = 0.108` vs
   `actual_std = 0.0042`(약 26배). 커버리지로 보면 **NAV가 원인**이다(NAV NMPIW95 = 1.86,
   PICP50 = 0.74 vs 명목 0.50). Combined는 이를 상속한다. NAV 스케일 조정(`scale_factor`)
   로직 점검 필요. **현재 최우선 항목.**
5. **`twad = inf`** — NAV/GAP/KP 전 모듈에서 무한대 (Combined만 0.003). 지표 정의상 분모 0 가능성.
6. **KS 검정 전 모듈 기각** — `ks_pvalue ~ 1e-33` 수준. 대표경로(median) vs 실제 비교라 당연한 결과일 수 있으나 논문 서술 시 해석 필요.

---

## 5. 커밋 이력 요약

| 커밋 | 날짜 | 내용 |
|---|---|---|
| `0c30cc2` ~ `b20f2e4` | 초기 | 초기 세팅, dataset, Poisson-GARCH NAV 시뮬레이터 |
| `68cd240` ~ `159a75d` | 2026-01-29 | |
| `a07b2f2` | 2026-02-12 | |
| `11239a5`, `415bd8b` | 2026-04-22 | |
| `9f2f1d3` | 2026-05-08 | |
| `2a66fb1` | 2025-05-15 | |
| `44ae59a` | 2026-05-31 | |
| `8644688` ~ `bf63c9f` | 2026-06-19 | 4회 커밋 |
| `7795358`, `0ac067d` | 2026-06-26 | |
| `f14c00a` | 2026-06-26 | **시나리오별 모수 재추정 버전** |
| `df14167` | 2026-06-26 | **모수 고정 버전** |
| `8b55590` | 2026-07-14 | `results_main.py` 추가 (유의성 검정), `wmcr_test*.py` 삭제 |
| `eb97732` | 2026-08-29 | Log Return 주말 손실 수정 (필터 -> 로그차분 순서) |
| `8da0e0e` | 2026-08-29 | 타 PC 인수인계 체크리스트 (3.1) |
| (작업중) | 2026-08-29 | **WMCR 전면 제거 -> 예측구간 커버리지(PICP/NMPIW) 대체** |

---

## 6. 회신 요청 사항

아래 항목에 답을 주시면 그 기준으로 정리·삭제를 진행하겠습니다.

- [ ] **4.1** 깨진 import 파일 — 복구 / 삭제 ?
- [ ] **4.2** 미사용 코드 각각 — 삭제 / 보존 ?
- [ ] **4.3** 최종 데이터셋 — `train` / `train_ver1` / 이원화 유지 ?
- [ ] **2.7** 시나리오 모수 — 재추정(`para_reestimate`) / 고정(`fixed_parameter_ver`) 중 최종본은 ?
- [ ] **4.4-1** 나머지 시나리오(S01~S04, S06~S09) 시뮬레이션 실행 필요 ?
- [ ] **4.4-2~6** 결과 이상 항목 — 수정 필요 / 논문 미포함이라 무시 ?
