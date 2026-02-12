# -*- coding: utf-8 -*-
"""
kp_variables.csv: KOSPI_Volatility vs Volume 상관관계 분석
- 데이터: dataset/kp_variables.csv
- Volume = volume_btc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CORR_DIR = Path(__file__).resolve().parent
DATASET_DIR = CORR_DIR.parent / "dataset"
KP_PATH = DATASET_DIR / "kp_variables.csv"

COL_X = "KOSPI_Volatility"
COL_Y = "volume_btc"  # Volume


def main():
    df = pd.read_csv(KP_PATH, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"])
    # 0 또는 결측 제거 (KOSPI_Volatility 0.0 등)
    df = df.dropna(subset=[COL_X, COL_Y])
    df = df[(df[COL_X] != 0) | (df[COL_Y] != 0)]

    x, y = df[COL_X].values, df[COL_Y].values
    pearson = np.corrcoef(x, y)[0, 1]
    spearman = df[COL_X].corr(df[COL_Y], method="spearman")

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-9) * 100
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-9) * 100
    rmse = np.sqrt(np.mean((x_n - y_n) ** 2))
    mae = np.mean(np.abs(x_n - y_n))

    window = min(30, len(df) // 3)
    df = df.sort_values("Date").reset_index(drop=True)
    df["rolling_corr"] = df[COL_X].rolling(window).corr(df[COL_Y])

    print("=" * 60)
    print("KP: KOSPI_Volatility vs Volume (volume_btc)")
    print("=" * 60)
    print(f"데이터: {KP_PATH}")
    print(f"일수: {len(df)} ({df['Date'].min()} ~ {df['Date'].max()})")
    print(f"\nPearson:  {pearson:.4f}")
    print(f"Spearman: {spearman:.4f}")
    print(f"정규화 RMSE(0~100): {rmse:.2f}, MAE: {mae:.2f}")
    print(f"롤링({window}일) 상관 평균: {df['rolling_corr'].mean():.4f}")

    if abs(pearson) >= 0.7:
        judgment = "강한 상관 (양/음)"
    elif abs(pearson) >= 0.5:
        judgment = "중간 정도 상관"
    elif abs(pearson) >= 0.3:
        judgment = "약한 상관"
    else:
        judgment = "매우 약하거나 무상관"
    print(f"판단: {judgment}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(df["Date"], df[COL_X], label=COL_X, alpha=0.8)
    ax1.set_ylabel(COL_X)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_title("KOSPI_Volatility 시계열")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax1b = ax1.twinx()
    ax1b.plot(df["Date"], df[COL_Y], color="orange", label="Volume (volume_btc)", alpha=0.8)
    ax1b.set_ylabel("Volume (volume_btc)")
    ax1b.legend(loc="upper right")

    ax2 = axes[0, 1]
    ax2.scatter(df[COL_X], df[COL_Y], alpha=0.5, s=10)
    ax2.set_xlabel(COL_X)
    ax2.set_ylabel("Volume (volume_btc)")
    ax2.set_title(f"산점도 (Pearson={pearson:.3f}, Spearman={spearman:.3f})")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[1, 0]
    valid = df.dropna(subset=["rolling_corr"])
    ax3.plot(valid["Date"], valid["rolling_corr"], color="green", alpha=0.8)
    ax3.axhline(y=pearson, color="gray", linestyle="--", label=f"Pearson={pearson:.3f}")
    ax3.set_title(f"롤링 상관계수 ({window}일)")
    ax3.set_ylabel("상관계수")
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    ax4 = axes[1, 1]
    ax4.plot(df["Date"], x_n, label=f"{COL_X} (정규화)", alpha=0.8)
    ax4.plot(df["Date"], y_n, label="Volume (정규화)", alpha=0.8)
    ax4.set_title("정규화(0~100) 시계열 비교")
    ax4.legend()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    out_path = CORR_DIR / "kp_correlation_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n시각화 저장: {out_path}")

    return df


if __name__ == "__main__":
    main()
