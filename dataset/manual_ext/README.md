# 수동 다운로드가 필요한 데이터 (확장 표본 2026-06-30까지)

자동 수집이 불가능한 두 개만 여기에 넣어주면 된다.
자동화를 시도했으나 아래와 같이 모두 막혀 있었다.

| 대상 | 시도 | 결과 |
|---|---|---|
| VKOSPI | KRX `getJsonData.cmd` (세션 쿠키 포함) | 지수 finder 전체(KOSPI 171개 포함)에 변동성지수 자체가 없음 |
| VKOSPI | pykrx | KRX 응답이 JSON이 아님 (차단) |
| VKOSPI | 네이버 `siseJson.naver` | KOSPI/KPI200은 되지만 VKOSPI 심볼 미지원 |
| VKOSPI | investing.com | HTTP 403 (봇 차단) |
| IBIT NAV | iShares `*.ajax?fileType=csv` (id 4종) | 전부 SPA HTML 반환 — CSV 엔드포인트 폐지됨 |
| IBIT NAV | stockanalysis.com | HTTP 403 |

VKOSPI에 한해 `--kospi-proxy` 옵션으로 **^KS11 30일 실현변동성(연환산 %)** 대용치를 쓸 수 있다.
다만 실제 내재변동성 지수가 아니므로 기존 결과와 비교하려면 아래 실물 파일을 넣는 편이 낫다.

```
etf/Scripts/python.exe dataset/collect_extended.py --kospi-proxy
```

파일을 넣은 뒤 `etf/Scripts/python.exe dataset/collect_extended.py`를 다시 실행하면
나머지는 이미 캐시되어 있으므로 수십 초 안에 해당 컬럼이 채워진다.

---

## 1. `KOSPI_Volatility.csv` — 코스피200 변동성지수 (VKOSPI)

**필요 기간**: 2022-05-01 ~ 2026-06-30
**쓰이는 곳**: `kp_variables.csv`의 `KOSPI_Volatility`, `train_ext/kp_train_main.csv`,
`train_ver1_ext/kp_train.csv` (KP 모형의 핵심 외생변수)

**받는 곳 (둘 중 하나)**

- investing.com → "코스피 변동성" 과거 데이터 → 기간 지정 후 다운로드
  (기존 `dataset/raw/KOSPI Volatility_history.csv`와 같은 경로)
- KRX 정보데이터시스템 `data.krx.co.kr` → 지수 > 주가지수 > 전체지수 시세추이
  → "코스피 200 변동성지수" 선택 → CSV 다운로드

**요구 포맷**: 앞의 두 컬럼만 읽는다. `날짜, 종가` 순서면 된다.

```
날짜,종가
2026-06-30,18.42
2026-06-29,18.77
```

날짜는 `2026-06-30` / `2026- 06- 30` / `2026/06/30` 모두 인식한다.
`2026년 06월 30일` 형태로 받아졌다면 그대로 둬도 되지만, 파싱 실패 시 알려주면 파서를 맞춰주겠다.

---

## 2. `IBIT_NAV.csv` — iShares Bitcoin Trust 일별 NAV

**필요 기간**: 2024-01-12 ~ 2026-06-30
**쓰이는 곳**: `y_true_variables.csv`의 `nav_true`, 그리고 이것으로 계산되는
`y_variables.csv`의 `etf_premium` (= `etf_true / nav_true - 1`)
→ **이 파일이 없으면 ETF 프리미엄 계열 전체가 비어 있다.**

**받는 곳**: iShares IBIT 상품 페이지 (ishares.com, 티커 IBIT)
→ Performance / NAV 섹션의 과거 NAV 다운로드 (Historical NAV, CSV)

**요구 포맷**: 앞의 두 컬럼만 읽는다. `날짜, NAV` 순서면 된다.

```
As Of,NAV per Share
2026-06-30,58.11
2026-06-27,57.84
```

`$` 기호나 천단위 쉼표는 자동으로 제거한다.

---

## 참고: 자동 수집되는 나머지

blockchain.info(온체인 6종) · Upbit(거래량/거래대금/BTC_KRW) · Binance(BTC_USD) ·
FRED DEXKOUS(USD_KRW) · yfinance(^VIX, IBIT 종가) · Google Trends(Bitcoin/비트코인)
는 `dataset/collect_extended.py`가 전부 받아온다. 재실행해도 `raw_ext/_cache/`에
캐시되어 있어 네트워크를 다시 타지 않는다.
