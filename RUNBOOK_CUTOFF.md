# 절단 실험 실행 안내 (다른 머신용)

지정학적 충격(2026-02 이후 VKOSPI 급등)이 부호 반전의 원인인지 확인하기 위해
확장 데이터를 두 시점에서 절단해 전체 파이프라인을 다시 돌린다.

| 작업공간 | 절단일 | 일수 | VKOSPI 최대 | 성격 |
|---|---|---|---|---|
| `ETF_code_ext_2512` | 2025-12-31 | 493 | 45.86 | 충격 이전 (VKOSPI>50 최초일 = 2026-02-03) |
| `ETF_code_ext_2603` | 2026-03-31 | 554 | 80.37 | 충격 초입 |
| `ETF_code_ext` | full | 617 | 96.94 | 충격 정점 포함 (이미 실행 완료) |

## 1. 환경

Python 3.11. 리포에 가상환경은 없다(`etf/` gitignore).

```
git clone <repo> ETF_code
cd ETF_code
python -m venv etf
etf\Scripts\pip install -r requirements.txt        # Windows
# source etf/bin/activate && pip install -r requirements.txt   # Linux/mac
```

`requirements.txt`의 앞 11개만 있으면 파이프라인이 돈다. 뒤 3개(pytrends, pyupbit,
yfinance)는 데이터 재수집 전용이라 설치 실패해도 무방하다.

## 2. 작업공간 생성

리포 밖에 격리된 사본을 만든다. 데이터는 `dataset/*_ext`(git 추적)에서 가져오므로
네트워크가 필요 없다.

```
etf\Scripts\python dataset\build_cutoff_workspace.py 2025-12-31 ..\ETF_code_ext_2512
etf\Scripts\python dataset\build_cutoff_workspace.py 2026-03-31 ..\ETF_code_ext_2603
```

`full`을 주면 확장판 전체, `--ctrl`을 붙이면 대조군(트렌드 기존값 보존) 데이터를 쓴다.

## 3. 실행

```
etf\Scripts\python -u dataset\run_cutoff_pipeline.py ..\ETF_code_ext_2512 ..\ETF_code_ext_2603
```

작업공간당 6단계를 순차 실행한다: Phase 1 → HMM 레짐 분류 → 시나리오 선정 →
기존 9개와 자동 매칭 → 시나리오 시뮬레이션 → 유의성 검정(M=100).
`--M 300`으로 반복 횟수를 바꿀 수 있다.

**소요 시간**: 작업공간당 약 2~2.5시간, 둘 합쳐 4~5시간. HMM 부트스트랩이
`joblib`로 코어를 전부 쓴다.

**분리 실행** (터미널을 닫아도 계속):

```
# Windows PowerShell
Start-Process etf\Scripts\python.exe -ArgumentList "-u","dataset\run_cutoff_pipeline.py","..\ETF_code_ext_2512","..\ETF_code_ext_2603" -WindowStyle Hidden -RedirectStandardOutput ..\cutoff.out -RedirectStandardError ..\cutoff.err

# Linux/mac
nohup etf/bin/python -u dataset/run_cutoff_pipeline.py ../ETF_code_ext_2512 ../ETF_code_ext_2603 > ../cutoff.out 2>&1 &
```

**절전 주의**: Windows는 실행 전 `powercfg /change standby-timeout-ac 0`으로 꺼야
한다. 스크립트가 끝나면 5분으로 되돌린다. Linux/mac은 각자 설정.

## 4. 진행 확인

```
Get-Content ..\cutoff_pipeline.log            # 단계별 시작/종료/소요시간/매칭 ID
Get-Content ..\ETF_code_ext_2512\run_results.log -Tail 20   # 개별 단계 상세
```

## 5. 결과 위치

```
<작업공간>/results/simulator/validation_results.json                     Phase 1 검증·모수
<작업공간>/results/regime_classified_table.csv                            레짐 조합
<작업공간>/results/scenario_selection/final_scenarios_latest.csv          선정 시나리오
<작업공간>/results/scenario_selection/match_table.csv                     기존 9개 대응표
<작업공간>/results/scenario_simulator/significance_test/*_kr.csv          효과 크기·p값
```

## 6. 확인할 것

`significance_test/effect_size_matrix_pct_kr.csv`의 `Volatility` 행 부호.

| 버전 | 예상 |
|---|---|
| 기존 342일 | 9개 전부 양(+) |
| 2512 (충격 이전) | 양(+)이 유지되면 → 부호 반전은 지정학적 충격 때문 |
| 2603 (충격 초입) | 혼재하면 → 충격 진입과 함께 반전 시작 |
| 2606 (정점 포함) | 17개 중 3개만 양(+) — 확인됨 |

주의: 절단마다 축 해상도(2수준/3수준)가 다르게 나올 수 있다. `match_table.csv`의
`exact` 열로 정확 일치 여부를 보고, 대응표 단위로 비교해야 한다.
