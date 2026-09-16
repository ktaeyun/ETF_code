"""Build compact, publication-oriented figures for the Results section.

The script reads only finalized pipeline outputs. It does not refit models or
rerun simulations, so figures remain traceable to the reported CSV results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#3366A6"
RED = "#C44E52"
ORANGE = "#E69F00"
GRAY = "#4D4D4D"
LIGHT_BLUE = "#9ECAE1"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def baseline_fan_chart() -> None:
    df = pd.read_csv(ROOT / "results" / "simulator" / "korean_etf_price.csv")
    dates = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.fill_between(
        dates,
        df["simulated_p5_krw"],
        df["simulated_p95_krw"],
        color=LIGHT_BLUE,
        alpha=0.42,
        label="Simulated 5–95% interval",
    )
    ax.plot(dates, df["simulated_median_krw"], color=BLUE, lw=2.0, label="Simulated median")
    ax.plot(dates, df["actual_etf_krw"], color=GRAY, lw=1.45, label="Observed proxy")
    ax.set(title="Baseline simulation of the Korean Bitcoin ETF proxy", ylabel="Price (KRW)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.75)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    last = df.iloc[-1]
    note = (
        f"Terminal: proxy {last.actual_etf_krw:,.0f} | "
        f"median {last.simulated_median_krw:,.0f} | "
        f"5–95% {last.simulated_p5_krw:,.0f}–{last.simulated_p95_krw:,.0f} KRW"
    )
    ax.text(0.01, 0.025, note, transform=ax.transAxes, fontsize=9, color=GRAY)
    fig.tight_layout()
    save(fig, "figure_1_baseline_fan_chart.png")


def scenario_volatility() -> None:
    effects = pd.read_csv(
        ROOT / "results" / "scenario_simulator" / "significance_test" / "effect_size_matrix_pct_kr.csv",
        index_col="metric",
    )
    scenarios = pd.read_csv(ROOT / "results" / "scenario_selection" / "final_scenarios_latest.csv")
    values = effects.loc["Volatility"].rename("effect").rename_axis("Scenario_ID").reset_index()
    df = values.merge(scenarios, on="Scenario_ID").sort_values("effect")
    high = df["KR_Volume"].eq("High") & df["KR_SVI"].eq("High")
    colors = np.where(high, RED, BLUE)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.barh(df["Scenario_ID"], df["effect"], color=colors, height=0.68)
    for bar, value in zip(bars, df["effect"]):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    ax.set(
        title="Scenario effect on Korean-factor volatility",
        xlabel="Effect relative to baseline (%)",
        ylabel="Scenario",
        xlim=(0, 27),
    )
    ax.grid(axis="x", color="#D9D9D9", lw=0.7, alpha=0.75)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=RED, label="KR volume & search: High"), Patch(color=BLUE, label="KR volume & search: Low")],
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    save(fig, "figure_2_scenario_volatility.png")


def effect_heatmaps() -> None:
    base = ROOT / "results" / "scenario_simulator" / "significance_test"
    etf = pd.read_csv(base / "effect_size_matrix_pct_etf.csv", index_col="metric")
    kr = pd.read_csv(base / "effect_size_matrix_pct_kr.csv", index_col="metric")
    order = ["Volatility", "Max_DD_mean", "VaR_99", "CVaR_99", "VaR_95", "CVaR_95", "p50", "p95"]
    labels = ["Volatility", "Max drawdown", "VaR 99", "CVaR 99", "VaR 95", "CVaR 95", "Median", "Upper 5%"]
    etf, kr = etf.loc[order], kr.loc[order]
    vmax = max(abs(etf.to_numpy()).max(), abs(kr.to_numpy()).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, data, title in zip(axes, [etf, kr], ["A. Full ETF price", "B. Korean factors only"]):
        image = ax.imshow(data.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(data.shape[1]), data.columns)
        ax.set_yticks(range(data.shape[0]), labels)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data.iat[row, col]
                text_color = "white" if abs(value) > vmax * 0.48 else "#222222"
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7.8, color=text_color)
        ax.set_xlabel("Scenario")
        ax.tick_params(length=0)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.88, wspace=0.11)
    cax = fig.add_axes([0.905, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Effect relative to baseline (%)")
    fig.suptitle("Scenario effects depend on the measurement basis", y=0.99, fontsize=13)
    save(fig, "figure_3_effect_heatmaps.png")


if __name__ == "__main__":
    baseline_fan_chart()
    scenario_volatility()
    effect_heatmaps()
    print(f"Saved figures to {OUT}")
