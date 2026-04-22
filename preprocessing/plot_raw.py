"""
Raw 데이터 시계열 시각화
========================
dataset/train_ver1/ 의 5개 원시 변수를 line graph 로 시각화한다.

변수:
  1. Global BTC SVI      (gap_train["value"],           Google Trends 글로벌, 0~100)
  2. Domestic BTC SVI    (kp_train["bitcoin_kr"],        Google Trends 국내,   0~100)
  3. BTC Volume (BTC)    (kp_train["volume_btc"],        일별 거래량)
  4. VKOSPI              (kp_train["KOSPI_Volatility"],  KOSPI 변동성 지수)
  5. Global RV (BTC)     (gap_train["btc_volatility"],   BTC 일별 실현 변동성)
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DATA_DIR    = _ROOT / "dataset" / "train_ver1"
RESULTS_DIR = _ROOT / "results" / "preprocessing"


# ── 변수 메타 정보 ──────────────────────────────────────────
_SERIES_META = [
    {
        "key":    "global_btc_svi",
        "label":  "Global BTC SVI",
        "ylabel": "Search Interest (0–100)",
        "color":  "#2166ac",
        "src":    "gap_train[\"value\"]",
    },
    {
        "key":    "domestic_btc_svi",
        "label":  "Domestic BTC SVI (KR)",
        "ylabel": "Search Interest (0–100)",
        "color":  "#4dac26",
        "src":    "kp_train[\"bitcoin_kr\"]",
    },
    {
        "key":    "btc_volume_btc",
        "label":  "BTC Trading Volume (BTC)",
        "ylabel": "Volume (BTC)",
        "color":  "#d6604d",
        "src":    "kp_train[\"volume_btc\"]",
    },
    {
        "key":    "VKOSPI",
        "label":  "VKOSPI",
        "ylabel": "Volatility Index",
        "color":  "#7b2d8b",
        "src":    "kp_train[\"KOSPI_Volatility\"]",
    },
    {
        "key":    "Global_RV",
        "label":  "Global BTC Realized Volatility",
        "ylabel": "Realized Volatility",
        "color":  "#f4a736",
        "src":    "gap_train[\"btc_volatility\"]",
    },
]


def load_raw() -> pd.DataFrame:
    """5개 변수를 일별 DataFrame 으로 로드."""
    kp_df  = pd.read_csv(DATA_DIR / "kp_train.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index()
    gap_df = pd.read_csv(DATA_DIR / "gap_train.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index()

    return pd.DataFrame({
        "global_btc_svi":   gap_df["value"],
        "domestic_btc_svi": kp_df["bitcoin_kr"],
        "btc_volume_btc":   kp_df["volume_btc"],
        "VKOSPI":           kp_df["KOSPI_Volatility"],
        "Global_RV":        gap_df["btc_volatility"],
    }).sort_index()


def plot_raw_data(
    df:        pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """5개 원시 변수 line graph.

    Parameters
    ----------
    df        : load_raw() 반환 DataFrame
    save_path : PNG 저장 경로 (None 이면 저장 안 함)
    """
    n   = len(_SERIES_META)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)

    date_fmt = mdates.DateFormatter("%Y-%m")
    date_loc = mdates.MonthLocator(interval=3)

    for ax, meta in zip(axes, _SERIES_META):
        key = meta["key"]
        if key not in df.columns:
            ax.set_visible(False)
            continue

        s = df[key].dropna()

        ax.plot(s.index, s.values,
                color=meta["color"], linewidth=0.9, alpha=0.9)

        # 최솟값·최댓값 마커
        ax.axhline(y=s.mean(), color="gray", linewidth=0.7,
                   linestyle="--", alpha=0.6, label=f"Mean = {s.mean():.2f}")

        ax.set_ylabel(meta["ylabel"], fontsize=9)
        ax.set_title(
            f"{meta['label']}  "
            f"[min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}  "
            f"std={s.std():.2f}]",
            fontsize=10, loc="left",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.2f}")
        )
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_loc)

    # x축 공유 레이블
    axes[-1].set_xlabel("Date", fontsize=10)
    fig.autofmt_xdate(rotation=30, ha="right")

    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end   = df.index.max().strftime("%Y-%m-%d")
    fig.suptitle(
        f"Raw Data Time Series  |  {date_start} ~ {date_end}  (N={len(df)})",
        fontsize=13, fontweight="bold", y=1.005,
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")

    return fig


if __name__ == "__main__":
    df = load_raw()

    print(f"Dataset : {df.shape[0]} rows  "
          f"({df.index[0].date()} ~ {df.index[-1].date()})")
    print("\nDescriptive Statistics:")
    print(df.describe().T.to_string())

    save_path = str(RESULTS_DIR / "raw_data_timeseries.png")
    fig = plot_raw_data(df, save_path=save_path)
    plt.show()
