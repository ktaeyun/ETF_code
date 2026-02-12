# -*- coding: utf-8 -*-
"""
bitcoin_kr / bitcoin_all 트렌드 지수 상관관계 분석
- 데이터: trend 폴더 (bitcoin_kr_*.csv, bitcoin_all_*.csv)
- 시간순 병합 후 Pearson/Spearman, 시각화, 유사성 판단
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CORR_DIR = Path(__file__).resolve().parent
TREND_DIR = CORR_DIR.parent / "trend"  # 데이터는 trend 폴더


def load_single_trend_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2, encoding="utf-8")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "value"])
    return df[["date", "value"]]


def load_and_merge_pattern(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(TREND_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No files: {TREND_DIR / pattern}")
    frames = [load_single_trend_csv(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return merged


def main():
    df_kr = load_and_merge_pattern("bitcoin_kr_*.csv")
    df_all = load_and_merge_pattern("bitcoin_all_*.csv")

    print("=" * 60)
    print("1. 병합된 시계열 (시간순, 데이터: trend 폴더)")
    print("=" * 60)
    print(f"bitcoin_kr (한국): {len(df_kr)}일, {df_kr['date'].min()} ~ {df_kr['date'].max()}")
    print(f"bitcoin_all (전세계): {len(df_all)}일, {df_all['date'].min()} ~ {df_all['date'].max()}")

    common = pd.merge(
        df_kr.rename(columns={"value": "kr"}),
        df_all.rename(columns={"value": "all"}),
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    print(f"\n공통 일수: {len(common)}일")

    kr, all_ = common["kr"].values, common["all"].values
    pearson = np.corrcoef(kr, all_)[0, 1]
    spearman = pd.Series(kr).corr(pd.Series(all_), method="spearman")

    kr_n = (kr - kr.min()) / (kr.max() - kr.min() + 1e-9) * 100
    all_n = (all_ - all_.min()) / (all_.max() - all_.min() + 1e-9) * 100
    rmse = np.sqrt(np.mean((kr_n - all_n) ** 2))
    window = min(30, len(common) // 3)
    common_roll = common.copy()
    common_roll["rolling_corr"] = common_roll["kr"].rolling(window).corr(common_roll["all"])

    print("\n3. 정량적 상관/유사도")
    print(f"  Pearson: {pearson:.4f}, Spearman: {spearman:.4f}, RMSE(0~100): {rmse:.2f}")

    if pearson >= 0.85 and spearman >= 0.85:
        judgment = "유사함"
    elif pearson >= 0.7 or spearman >= 0.7:
        judgment = "다소 유사함"
    elif pearson >= 0.5 or spearman >= 0.5:
        judgment = "약한 유사성"
    else:
        judgment = "유사하지 않음"
    print(f"  판단: {judgment}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1 = axes[0, 0]
    ax1.plot(common["date"], common["kr"], label="한국 (bitcoin_kr)", alpha=0.8)
    ax1.plot(common["date"], common["all"], label="전세계 (bitcoin_all)", alpha=0.8)
    ax1.set_title("한국 vs 전세계 검색 트렌드 (공통 기간)")
    ax1.set_ylabel("트렌드 값")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[0, 1]
    ax2.scatter(common["kr"], common["all"], alpha=0.5, s=10)
    ax2.set_xlabel("한국 (bitcoin_kr)")
    ax2.set_ylabel("전세계 (bitcoin_all)")
    ax2.set_title(f"산점도 (Pearson={pearson:.3f}, Spearman={spearman:.3f})")
    mx = max(common["kr"].max(), common["all"].max())
    ax2.plot([0, mx], [0, mx], "k--", alpha=0.5, label="y=x")
    ax2.legend()

    ax3 = axes[1, 0]
    valid = common_roll.dropna(subset=["rolling_corr"])
    ax3.plot(valid["date"], valid["rolling_corr"], color="green", alpha=0.8)
    ax3.axhline(y=pearson, color="gray", linestyle="--", label=f"Pearson={pearson:.3f}")
    ax3.set_title(f"롤링 상관계수 ({window}일)")
    ax3.set_ylabel("상관계수")
    ax3.legend()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    ax4 = axes[1, 1]
    ax4.plot(common["date"], kr_n, label="한국 (정규화)", alpha=0.8)
    ax4.plot(common["date"], all_n, label="전세계 (정규화)", alpha=0.8)
    ax4.set_title("정규화(0~100) 시계열 비교")
    ax4.legend()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    out_path = CORR_DIR / "trend_correlation_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n시각화 저장: {out_path}")

    # 병합 결과는 데이터이므로 trend 폴더에 저장
    df_kr.to_csv(TREND_DIR / "merged_bitcoin_kr.csv", index=False, encoding="utf-8-sig")
    df_all.to_csv(TREND_DIR / "merged_bitcoin_all.csv", index=False, encoding="utf-8-sig")
    common.to_csv(TREND_DIR / "merged_common_for_correlation.csv", index=False, encoding="utf-8-sig")

    return df_kr, df_all, common


if __name__ == "__main__":
    main()
