# -*- coding: utf-8 -*-
"""
4개 데이터 각각 시각화
1. 트렌드 (bitcoin_kr, bitcoin_all) - trend 폴더
2. nav_variables.csv
3. kp_variables.csv
4. y_variables.csv
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

PLOT_DIR = Path(__file__).resolve().parent
TREND_DIR = PLOT_DIR.parent / "trend"
DATASET_DIR = PLOT_DIR.parent / "dataset"


def load_trend_merged():
    """trend 폴더에서 병합된 트렌드 로드 (또는 원본 병합)"""
    def load_one(path):
        df = pd.read_csv(path, skiprows=2, encoding="utf-8")
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "value"]].dropna()
    kr_files = sorted(glob.glob(str(TREND_DIR / "bitcoin_kr_*.csv")))
    all_files = sorted(glob.glob(str(TREND_DIR / "bitcoin_all_*.csv")))
    if not kr_files or not all_files:
        raise FileNotFoundError("trend/*.csv not found")
    df_kr = pd.concat([load_one(f) for f in kr_files]).drop_duplicates(subset=["date"]).sort_values("date")
    df_all = pd.concat([load_one(f) for f in all_files]).drop_duplicates(subset=["date"]).sort_values("date")
    return df_kr.rename(columns={"value": "bitcoin_kr"}), df_all.rename(columns={"value": "bitcoin_all"})


def plot_trend():
    df_kr, df_all = load_trend_merged()
    common = pd.merge(df_kr, df_all, on="date", how="inner").sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(common["date"], common["bitcoin_kr"], label="bitcoin_kr (한국)", alpha=0.8)
    ax.plot(common["date"], common["bitcoin_all"], label="bitcoin_all (전세계)", alpha=0.8)
    ax.set_title("트렌드 지수: 한국 vs 전세계 (공통 기간)")
    ax.set_ylabel("트렌드 값")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "1_trend.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: plot/1_trend.png")


def plot_nav():
    path = DATASET_DIR / "nav_variables.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Date"])

    cols = [c for c in df.columns if c != "Date"]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].plot(df["Date"], df[col], alpha=0.8)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    fig.suptitle("nav_variables.csv 시계열", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "2_nav_variables.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: plot/2_nav_variables.png")


def plot_kp():
    path = DATASET_DIR / "kp_variables.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Date"])

    cols = [c for c in df.columns if c != "Date"]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].plot(df["Date"], df[col], alpha=0.8)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    fig.suptitle("kp_variables.csv 시계열", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "3_kp_variables.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: plot/3_kp_variables.png")


def plot_y():
    path = DATASET_DIR / "y_variables.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Date"])

    cols = [c for c in df.columns if c != "Date"]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        axes[i].plot(df["Date"], df[col], alpha=0.8)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    fig.suptitle("y_variables.csv 시계열", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "4_y_variables.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: plot/4_y_variables.png")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_trend()
    plot_nav()
    plot_kp()
    plot_y()
    print("Done. 4 plots in plot/")


if __name__ == "__main__":
    main()
