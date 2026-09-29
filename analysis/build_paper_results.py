# -*- coding: utf-8 -*-
"""
절단 작업공간(예: ../ETF_code_ext_2512)의 최종 산출물만 읽어 논문 Results 절용
표(CSV + Markdown)와 그림(PNG)을 한 폴더에 만든다. 모형 재적합·재시뮬레이션은 하지 않는다
(NAV ARIMA-GARCH 모수는 캐시에 저장되지 않아 표 작성용으로만 빠르게 재적합한다).

    python analysis/build_paper_results.py <작업공간> <출력폴더>
    python analysis/build_paper_results.py ../ETF_code_ext_2512 results/paper_2512

읽는 파일 (작업공간 기준)
    results/simulator/validation_results.json, cache/sim_meta.json, cache/sim_arrays.npz
    results/simulator/korean_etf_price.csv, gap_simulation_results.csv, kp_simulation_results.csv
    results/scenario_selection/final_scenarios_latest.csv, match_table.csv, matched_ids.txt
    results/regime_classified_table.csv
    results/scenario_simulator/significance_test/*.csv, korean_etf_risk_metrics.csv
    run_preproc.log (HMM K 선택 근거), dataset/raw, dataset/train (NAV 재적합용)
"""
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ── 팔레트 (dataviz 기본 팔레트, 검증 통과 순서) ────────────────────────
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, VIOLET = "#e34948", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#0d366b", BLUE, "#f0efec", RED, "#7a1f1f"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b3", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 130, "savefig.dpi": 300,
    "axes.titlecolor": INK, "legend.frameon": False,
})

AX = ["Bitcoin_RV", "VKOSPI", "KR_Volume", "KR_SVI", "Global_SVI"]
METRICS = ["Volatility", "Max_DD_mean", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99", "p50", "p95"]
METRIC_LABEL = {"Volatility": "Volatility", "Max_DD_mean": "Max drawdown", "VaR_95": "VaR 95",
                "CVaR_95": "CVaR 95", "VaR_99": "VaR 99", "CVaR_99": "CVaR 99",
                "p50": "Median (p50)", "p95": "Upper 5% (p95)"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════
# 1. 로드
# ══════════════════════════════════════════════════════════════
class WS:
    def __init__(self, root: Path, use_all: bool = False):
        self.root = root
        R = root / "results"
        self.val = json.loads((R / "simulator" / "validation_results.json").read_text(encoding="utf-8"))
        self.meta = json.loads((R / "simulator" / "cache" / "sim_meta.json").read_text(encoding="utf-8"))
        self.arr = np.load(R / "simulator" / "cache" / "sim_arrays.npz")
        self.price = pd.read_csv(R / "simulator" / "korean_etf_price.csv")
        self.price["Date"] = pd.to_datetime(self.price["Date"])
        self.gap_sim = pd.read_csv(R / "simulator" / "gap_simulation_results.csv")
        self.kp_sim = pd.read_csv(R / "simulator" / "kp_simulation_results.csv")
        self.nav_sim = pd.read_csv(R / "simulator" / "nav_simulation_results.csv")

        sel = R / "scenario_selection"
        self.scen = pd.read_csv(sel / "final_scenarios_latest.csv")
        ids_file = sel / "matched_ids.txt"
        self.ids = ids_file.read_text(encoding="utf-8").strip().split(",") if ids_file.exists() else None
        self.match = pd.read_csv(sel / "match_table.csv") if (sel / "match_table.csv").exists() else None
        self.table = pd.read_csv(R / "regime_classified_table.csv")

        st = R / "scenario_simulator" / "significance_test"
        rd = lambda n: pd.read_csv(st / n, index_col=0)
        self.eff = {b: rd(f"effect_size_matrix_pct_{b}.csv") for b in ("kr", "etf")}
        self.pval = {b: rd(f"significance_matrix_pvalue_{b}.csv") for b in ("kr", "etf")}
        self.flag = {b: rd(f"significance_matrix_flag_{b}.csv") for b in ("kr", "etf")}
        self.sig = {b: pd.read_csv(st / f"significance_results_{b}.csv") for b in ("kr", "etf")}
        self.raw = {b: pd.read_csv(st / f"raw_repeat_metrics_{b}.csv") for b in ("kr", "etf")}
        self.cmp = pd.read_csv(st / "effect_size_comparison.csv")
        self.risk = pd.read_csv(R / "scenario_simulator" / "korean_etf_risk_metrics.csv")
        self.preproc_log = root / "run_preproc.log"

        # 시나리오 순서: 매칭된 ID만, Volatility(kr) 효과 내림차순
        vol = self.eff["kr"].loc["Volatility"]
        if self.ids is None or use_all:
            self.ids = list(vol.index)
        self.ids = sorted([i for i in self.ids if i in vol.index], key=lambda i: -vol[i])
        self.scen = self.scen.set_index("Scenario_ID")

    def combo(self, sid):
        return self.scen.loc[sid, "combo_label"]


# ══════════════════════════════════════════════════════════════
# 2. 표
# ══════════════════════════════════════════════════════════════
def nav_params(ws: WS) -> dict:
    """NAV ARIMA(1,0,1)-GARCH(1,1)-t 모수는 캐시에 없어 재적합한다 (수 초)."""
    sys.path.insert(0, str(ws.root))
    from simulator.data_loader import load_nav_exog_and_returns
    from simulator.arima_garch_t_nav_simulator import fit_arimax_garch_t
    df = load_nav_exog_and_returns(base_dir=str(ws.root), nav_source=ws.meta.get("nav_source", "btc"))
    sim = fit_arimax_garch_t(log_returns=df["Log Return"], exog=None, ar_order=(1, 0, 1))
    ap = {k: float(v) for k, v in sim.arima_fitted.params.items()} if hasattr(sim.arima_fitted.params, "items") \
        else dict(zip(sim.arima_fitted.param_names, map(float, sim.arima_fitted.params)))
    gp = {k: float(v) for k, v in sim.garch_fitted.params.items()}
    out = {"T": int(len(df)), "start": str(df["Date"].iloc[0].date()), "end": str(df["Date"].iloc[-1].date())}
    out.update({f"arima_{k}": v for k, v in ap.items()})
    out.update({f"garch_{k}": v for k, v in gp.items()})
    return out


def table_params(ws: WS, nav: dict) -> pd.DataFrame:
    g = ws.val["gap"]["ou_params"]
    k = ws.val["kp"]["threshold_ou_params"]
    r0, r1 = k["regime_params"]["0"], k["regime_params"]["1"]
    rows = [
        ("NAV", "ARIMA(1,0,1)-GARCH(1,1)-t", "AR(1) φ", nav.get("arima_ar.L1", np.nan)),
        ("NAV", "", "MA(1) θ", nav.get("arima_ma.L1", np.nan)),
        ("NAV", "", "GARCH ω", nav.get("garch_omega", np.nan)),
        ("NAV", "", "GARCH α", nav.get("garch_alpha[1]", np.nan)),
        ("NAV", "", "GARCH β", nav.get("garch_beta[1]", np.nan)),
        ("NAV", "", "ν (Student-t)", nav.get("garch_nu", np.nan)),
        ("GAP", "OU-t (μ_t: Global SVI, σ_t: BTC RV)", "κ", g["kappa"]),
        ("GAP", "", "μ", g["mu"]),
        ("GAP", "", "σ₀", g["sigma0"]),
        ("GAP", "", "δ1 (Global SVI)", g["delta1"]),
        ("GAP", "", "δ2 (BTC RV)", g["delta2"]),
        ("GAP", "", "ν (Student-t)", g["nu"]),
        ("KP", "2-regime Threshold-OU-t", "τ (threshold)", k["threshold"]),
        ("KP", "", "ν (Student-t, common)", k["nu"]),
        ("KP", "Regime 0 (KP ≤ τ)", "κ₀", r0["kappa"]),
        ("KP", "", "μ₀", r0["mu"]),
        ("KP", "", "σ₀,₀", r0["sigma0"]),
        ("KP", "", "δ1₀ (KR volume)", r0["delta1"]),
        ("KP", "", "δ2₀ (VKOSPI)", r0["delta2"]),
        ("KP", "", "δ3₀ (KR SVI)", r0["delta3"]),
        ("KP", "Regime 1 (KP > τ)", "κ₁", r1["kappa"]),
        ("KP", "", "μ₁", r1["mu"]),
        ("KP", "", "σ₀,₁", r1["sigma0"]),
        ("KP", "", "δ1₁ (KR volume)", r1["delta1"]),
        ("KP", "", "δ2₁ (VKOSPI)", r1["delta2"]),
        ("KP", "", "δ3₁ (KR SVI)", r1["delta3"]),
    ]
    return pd.DataFrame(rows, columns=["component", "model", "parameter", "estimate"])


def table_validation(ws: WS) -> pd.DataFrame:
    names = [("nav", "NAV"), ("gap", "GAP"), ("kp", "KP"), ("us_etf", "US ETF (NAV×(1+GAP))"),
             ("combined", "Combined (NAV×(1+GAP)×(1+KP))")]
    rows = []
    for key, label in names:
        t = ws.val[key]["statistical_tests"]
        rows.append({
            "target": label,
            "PIT-KS p": t["pit_ks"]["ks_pvalue"],
            "Kupiec p": t["kupiec"]["pvalue"],
            "VaR exceedance (%)": 100 * t["kupiec"]["exceedance_rate"],
            "n_exceed": int(t["kupiec"]["n_exceedances"]),
            "ES p": t["es"]["pvalue"],
            "ES tail error": t["es"]["tail_error"],
            "ES percentile": t["es"]["percentile"],
            "verdict": "pass" if t["is_valid"] else "FAIL",
        })
    return pd.DataFrame(rows)


def table_hmm(ws: WS) -> pd.DataFrame:
    """run_preproc.log 의 'HMM Model Comparison' 블록을 파싱한다."""
    if not ws.preproc_log.exists():
        return pd.DataFrame()
    raw = ws.preproc_log.read_bytes()
    for enc in ("utf-8", "cp949"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            txt = raw.decode("utf-8", errors="replace")
    blocks = re.split(r"HMM Model Comparison\s*\|\s*Series:\s*", txt)[1:]
    rows = []
    for b in blocks:
        series = b.split("\n", 1)[0].strip()
        bic = re.search(r"BIC\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", b)
        p12 = re.search(r"K=1\s+vs\s+H1: K=2\):\s*\n.*?\n\s*p-value.*?:\s*([\d.]+)", b, re.S)
        p23 = re.search(r"K=2\s+vs\s+H1: K=3\):\s*\n.*?\n\s*p-value.*?:\s*([\d.]+)", b, re.S)
        kopt = re.search(r"Optimal K = (\d)", b)
        K = int(kopt.group(1)) if kopt else np.nan
        occ = re.findall(rf"K={K} Regime (\d): ([\d.]+)%", b) if kopt else []
        rows.append({
            "series": series,
            "BIC K=1": float(bic.group(1)) if bic else np.nan,
            "BIC K=2": float(bic.group(2)) if bic else np.nan,
            "BIC K=3": float(bic.group(3)) if bic else np.nan,
            "LRT p (1 vs 2)": float(p12.group(1)) if p12 else np.nan,
            "LRT p (2 vs 3)": float(p23.group(1)) if p23 else np.nan,
            "K": K,
            "occupancy (%)": " / ".join(f"{float(v):.1f}" for _, v in occ),
        })
    return pd.DataFrame(rows)


def table_scenarios(ws: WS) -> pd.DataFrame:
    rows = []
    for sid in ws.ids:
        s = ws.scen.loc[sid]
        orig, exact = "", ""
        if ws.match is not None:
            m = ws.match[ws.match["new"] == sid]
            orig = ", ".join(m["old"].tolist())
            exact = ", ".join(m.loc[m["exact"], "old"].tolist())
        rows.append({
            "Scenario_ID": sid, **{a: s[a] for a in AX}, "combo": s["combo_label"],
            "freq": int(s["joint_frequency"]), "lift": s["pairwise_lift_mean"],
            "matched_original": orig, "exact_match": exact,
            "Volatility_kr (%)": ws.eff["kr"].loc["Volatility", sid],
            "MaxDD_kr (%)": ws.eff["kr"].loc["Max_DD_mean", sid],
        })
    return pd.DataFrame(rows)


def table_effects(ws: WS, basis: str) -> pd.DataFrame:
    e, p = ws.eff[basis].loc[METRICS, ws.ids], ws.pval[basis].loc[METRICS, ws.ids]
    out = pd.DataFrame(index=[METRIC_LABEL[m] for m in METRICS], columns=ws.ids, dtype=object)
    for m in METRICS:
        for s in ws.ids:
            out.loc[METRIC_LABEL[m], s] = f"{e.loc[m, s]:+.3f}{stars(p.loc[m, s])}"
    out.index.name = "metric"
    return out.reset_index()


def table_sign_summary(ws: WS) -> pd.DataFrame:
    rows = []
    for basis in ("kr", "etf"):
        e, f = ws.eff[basis].loc[METRICS, ws.ids], ws.flag[basis].loc[METRICS, ws.ids]
        for m in METRICS:
            v = e.loc[m]
            rows.append({"basis": basis, "metric": METRIC_LABEL[m], "n_positive": int((v > 0).sum()),
                         "n_negative": int((v < 0).sum()), "n_significant": int(f.loc[m].astype(bool).sum()),
                         "min (%)": v.min(), "median_abs (%)": v.abs().median(), "max (%)": v.max()})
    return pd.DataFrame(rows)


def table_base_risk(ws: WS) -> pd.DataFrame:
    b = ws.risk[ws.risk["Scenario"] == "Base"].iloc[0]
    kr = ws.raw["kr"][ws.raw["kr"]["scenario_id"] == "Base"]
    etf = ws.raw["etf"][ws.raw["etf"]["scenario_id"] == "Base"]
    rows = [("ETF price (10,000 KRW anchor)", "VaR 95 (KRW)", b["VaR_95"]),
            ("", "CVaR 95 (KRW)", b["CVaR_95"]),
            ("", "VaR 99 (KRW)", b["VaR_99"]),
            ("", "CVaR 99 (KRW)", b["CVaR_99"]),
            ("", "Mean max drawdown", b["Max_DD_mean"]),
            ("", "Daily log-return volatility", etf["Volatility"].mean()),
            ("", "Median terminal price (KRW)", b["p50"]),
            ("", "p5 / p95 terminal (KRW)", f"{b['p5']:,.0f} / {b['p95']:,.0f}"),
            ("Korean factor (1+GAP)(1+KP)", "Daily log-return volatility", kr["Volatility"].mean()),
            ("", "Mean max drawdown", kr["Max_DD_mean"].mean()),
            ("", "Median terminal (KRW, 10,000 anchor)", kr["p50"].mean())]
    return pd.DataFrame(rows, columns=["basis", "metric", "Base"])


def table_group_means(ws: WS) -> pd.DataFrame:
    """Volatility(kr) 효과를 레짐 그룹별로 요약."""
    df = table_scenarios(ws)
    grp = np.where(df["VKOSPI"].eq("Extreme"), "VKOSPI Extreme",
                   np.where(df["KR_SVI"].eq("High") & df["KR_Volume"].isin(["High", "Mid"]),
                            "VKOSPI Normal & KR_SVI High", "Other (Normal, SVI Mid/Low)"))
    df["group"] = grp
    g = df.groupby("group")["Volatility_kr (%)"].agg(["count", "mean", "min", "max"]).reset_index()
    return g


# ══════════════════════════════════════════════════════════════
# 3. 그림
# ══════════════════════════════════════════════════════════════
def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelsize=8.5)


def save(fig, out: Path, name: str):
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_fanchart(ws: WS, out: Path):
    p = ws.price
    d = p["Date"]
    # Korean factor (1+GAP)(1+KP) paths, anchored 10,000
    mg, mk = ws.arr["mc_gap"], ws.arr["mc_kp"]
    T = min(mg.shape[1], mk.shape[1], len(d))
    raw = (1 + mg[:, :T]) * (1 + mk[:, :T])
    kr = raw * (10000.0 / raw[:, :1])
    ag, ak = ws.gap_sim["actual_gap"].to_numpy()[:T], ws.kp_sim["actual_kp"].to_numpy()[:T]
    kr_act = (1 + ag) * (1 + ak)
    kr_act = kr_act * (10000.0 / kr_act[0])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.fill_between(d, p["simulated_p5_krw"], p["simulated_p95_krw"], color=SEQ[1], alpha=0.5,
                    label="Simulated 5–95% band", lw=0)
    ax.plot(d, p["simulated_median_krw"], color=BLUE, lw=2, label="Simulated median")
    ax.plot(d, p["actual_etf_krw"], color=INK, lw=1.4, label="Observed proxy")
    ax.set(title="A. Korean Bitcoin ETF proxy (baseline)", ylabel="Price (KRW, anchored at 10,000)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    last = p.iloc[-1]
    ax.text(0.01, 0.97, f"Terminal — proxy {last.actual_etf_krw:,.0f} | median {last.simulated_median_krw:,.0f}"
            f" | band {last.simulated_p5_krw:,.0f}–{last.simulated_p95_krw:,.0f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92), fontsize=9)

    ax = axes[1]
    q5, q50, q95 = np.percentile(kr, [5, 50, 95], axis=0)
    ax.fill_between(d[:T], q5, q95, color=SEQ[1], alpha=0.5, lw=0, label="Simulated 5–95% band")
    ax.plot(d[:T], q50, color=BLUE, lw=2, label="Simulated median")
    ax.plot(d[:T], kr_act, color=INK, lw=1.4, label="Observed (1+GAP)(1+KP)")
    ax.set(title="B. Korean factor (1+GAP)(1+KP) only", ylabel="Index (anchored at 10,000)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="lower left", fontsize=9)
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.7)
        date_axis(a)
    fig.tight_layout()
    save(fig, out, "fig1_baseline_fanchart.png")


def fig_component_fit(ws: WS, out: Path):
    """GAP·KP 실측 vs 시뮬레이션 밴드 — 성분 모형 적합을 눈으로 확인."""
    d = ws.price["Date"]
    mg, mk = ws.arr["mc_gap"], ws.arr["mc_kp"]
    T = min(mg.shape[1], mk.shape[1], len(d))
    tau = ws.val["kp"]["threshold_ou_params"]["threshold"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, mc, act, name in [(axes[0], mg, ws.gap_sim["actual_gap"], "GAP (US ETF premium)"),
                              (axes[1], mk, ws.kp_sim["actual_kp"], "KP (Kimchi premium)")]:
        q5, q50, q95 = np.percentile(mc[:, :T] * 100, [5, 50, 95], axis=0)
        ax.fill_between(d[:T], q5, q95, color=SEQ[1], alpha=0.5, lw=0, label="Simulated 5–95% band")
        ax.plot(d[:T], q50, color=BLUE, lw=1.8, label="Simulated median")
        ax.plot(d[:T], act.to_numpy()[:T] * 100, color=INK, lw=1.1, label="Observed")
        ax.set(title=name, ylabel="%")
        ax.grid(axis="y", color=GRID, lw=0.7)
    axes[1].axhline(tau * 100, color=ORANGE, lw=1.4, ls="--")
    axes[1].text(d.iloc[-1], tau * 100, f"τ = {tau*100:.2f}% (regime threshold)  ", color=ORANGE,
                 va="bottom", ha="right", fontsize=9)
    share = float((ws.kp_sim["actual_kp"].to_numpy()[:T] > tau).mean() * 100)
    axes[1].text(0.99, 0.97, f"Observed share above τ: {share:.1f}%", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    date_axis(axes[1])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "fig2_gap_kp_fit.png")


def fig_validation(ws: WS, out: Path):
    v = table_validation(ws)
    tests = ["PIT-KS p", "Kupiec p", "ES p"]
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(v))[::-1]
    for j, (t, c) in enumerate(zip(tests, colors)):
        ax.scatter(v[t], y + (1 - j) * 0.22, s=70, color=c, label=t, zorder=3, edgecolor="white", lw=1)
        for yy, val in zip(y + (1 - j) * 0.22, v[t]):
            ax.text(val + 0.015, yy, f"{val:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0.05, color=RED, lw=1.2, ls="--")
    ax.text(0.052, len(v) - 0.45, "α = 0.05", color=RED, fontsize=8.5, va="top")
    ax.set_yticks(y, v["target"])
    ax.set(xlim=(0, 1.08), xlabel="p-value (larger = no evidence against the model)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
    ax.set_title("Goodness-of-fit tests: all targets pass at 5%", pad=26)
    fig.tight_layout()
    save(fig, out, "fig3_validation_pvalues.png")


def _group_color(row):
    if row["VKOSPI"] == "Extreme":
        return ORANGE, "VKOSPI Extreme"
    if row["KR_SVI"] == "High":
        return AQUA, "VKOSPI Normal, KR search interest High"
    return BLUE, "VKOSPI Normal, KR search interest Mid/Low"


def fig_volatility_bars(ws: WS, out: Path):
    df = table_scenarios(ws).sort_values("Volatility_kr (%)")
    p = ws.pval["kr"].loc["Volatility"]
    cols, labels = zip(*[_group_color(r) for _, r in df.iterrows()])
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ylab = [f"{s}  {c}" for s, c in zip(df["Scenario_ID"], df["combo"])]
    bars = ax.barh(ylab, df["Volatility_kr (%)"], color=cols, height=0.7)
    ax.axvline(0, color=INK, lw=1)
    for b, v, sid, fr in zip(bars, df["Volatility_kr (%)"], df["Scenario_ID"], df["freq"]):
        sig = "" if p[sid] < 0.05 else "  (n.s.)"
        ax.text(v + (0.3 if v >= 0 else -0.3), b.get_y() + b.get_height() / 2,
                f"{v:+.1f}%{sig}   n={fr}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=INK2)
    lim = max(abs(df["Volatility_kr (%)"])) * 1.35
    ax.set(xlim=(-lim, lim), xlabel="Korean-factor volatility, effect relative to baseline (%)",
           title="Scenario effect on Korean-factor volatility (kr basis)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(handles=[Patch(color=ORANGE, label="VKOSPI Extreme"),
                       Patch(color=AQUA, label="VKOSPI Normal, KR search interest High"),
                       Patch(color=BLUE, label="VKOSPI Normal, KR search interest Mid/Low")],
              loc="lower right", fontsize=9)
    ax.set_ylabel("Scenario  (Bitcoin RV – VKOSPI – KR volume – KR search – Global search)", fontsize=9)
    ax.text(0.01, 0.01, "n = observed days in the regime combination;  n.s. = not significant at 5%",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    fig.tight_layout()
    save(fig, out, "fig4_scenario_volatility.png")


def fig_heatmaps(ws: WS, out: Path):
    order = ["Volatility", "Max_DD_mean", "VaR_99", "CVaR_99", "VaR_95", "CVaR_95", "p50", "p95"]
    labels = [METRIC_LABEL[m] for m in order]
    etf, kr = ws.eff["etf"].loc[order, ws.ids], ws.eff["kr"].loc[order, ws.ids]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2))
    for ax, data, flag, title in zip(axes, [etf, kr], [ws.flag["etf"], ws.flag["kr"]],
                                     ["A. Full ETF price  NAV×(1+GAP)×(1+KP)", "B. Korean factors only  (1+GAP)×(1+KP)"]):
        vmax = float(abs(data.to_numpy()).max())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(data.to_numpy(), cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_title(title, loc="left")
        ax.set_xticks(range(data.shape[1]), data.columns, fontsize=9)
        ax.set_yticks(range(data.shape[0]), labels, fontsize=9)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.iat[i, j]
                sig = bool(flag.loc[order[i], data.columns[j]])
                tc = "white" if abs(v) > vmax * 0.5 else INK
                fmt = "{:.2f}" if vmax < 1.5 else "{:.1f}"
                ax.text(j, i, fmt.format(v) + ("" if sig else "°"), ha="center", va="center", fontsize=7.6, color=tc)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Effect vs baseline (%)", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    axes[1].set_xlabel("Scenario (sorted by Korean-factor volatility effect)")
    fig.text(0.01, 0.005, "° = not significant at 5% (paired test over M=100 CRN repeats). "
             "Negative Max drawdown/VaR/CVaR = larger downside risk.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, "fig5_effect_heatmaps.png")


def fig_vol_vs_dd(ws: WS, out: Path):
    df = table_scenarios(ws)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for _, r in df.iterrows():
        c, _ = _group_color(r)
        ax.scatter(r["Volatility_kr (%)"], r["MaxDD_kr (%)"], s=40 + r["freq"] * 3, color=c, alpha=0.85,
                   edgecolor="white", lw=1, zorder=3)
        ax.annotate(r["Scenario_ID"], (r["Volatility_kr (%)"], r["MaxDD_kr (%)"]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.9)
    ax.axvline(0, color=INK, lw=0.9)
    x, y = df["Volatility_kr (%)"], df["MaxDD_kr (%)"]
    r = np.corrcoef(x, y)[0, 1]
    ax.set(xlabel="Volatility effect (%)", ylabel="Mean max drawdown effect (%)  (+ = shallower)",
           title=f"Lower volatility goes with shallower drawdown (r = {r:.2f})")
    ax.grid(color=GRID, lw=0.7)
    ax.legend(handles=[Patch(color=ORANGE, label="VKOSPI Extreme"),
                       Patch(color=AQUA, label="Normal, KR search High"),
                       Patch(color=BLUE, label="Normal, KR search Mid/Low")], loc="upper right", fontsize=8.5)
    ax.text(0.01, 0.01, "Marker size ∝ observed days", transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, out, "fig6_volatility_vs_drawdown.png")


def fig_scenario_map(ws: WS, out: Path):
    """17개 시나리오의 레짐 구성 + 빈도 + 변동성 효과를 한 장에."""
    df = table_scenarios(ws)
    lv3 = {"Low": 0, "Mid": 1, "High": 2, "Normal": 0, "Extreme": 2}
    grid = np.array([[lv3[str(df.loc[i, a])] for a in AX] for i in df.index], dtype=float)
    n = len(df)
    fig = plt.figure(figsize=(12.5, 0.42 * n + 1.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 1.6, 2.6], wspace=0.06)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cmap = LinearSegmentedColormap.from_list("lv", [SEQ[0], SEQ[2], SEQ[5]])
    ax0.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(n):
        for j, a in enumerate(AX):
            v = str(df.loc[i, a])
            ax0.text(j, i, v, ha="center", va="center", fontsize=8.5,
                     color="white" if lv3[v] == 2 else INK)
    ax0.set_xticks(range(5), ["Bitcoin RV", "VKOSPI", "KR volume", "KR search", "Global search"], fontsize=9)
    ax0.xaxis.tick_top()
    ax0.set_yticks(range(n), [f"{s}  ({o})" if o else s for s, o in zip(df["Scenario_ID"], df["matched_original"])],
                   fontsize=8.5)
    ax0.tick_params(length=0)
    for sp in ax0.spines.values():
        sp.set_visible(False)
    ax0.set_ylabel("Scenario (original 2025-05 counterpart in parentheses)", fontsize=9)

    y = np.arange(n)
    ax1.barh(y, df["freq"], color=SEQ[3], height=0.7)
    for yy, v in zip(y, df["freq"]):
        ax1.text(v + 1, yy, str(v), va="center", fontsize=8, color=INK2)
    ax1.set(ylim=(n - 0.5, -0.5), yticks=[], title="Observed days")
    ax1.set_xlim(0, df["freq"].max() * 1.35)
    ax1.spines["left"].set_visible(False)

    vals = df["Volatility_kr (%)"]
    cols = [_group_color(r)[0] for _, r in df.iterrows()]
    ax2.barh(y, vals, color=cols, height=0.7)
    ax2.axvline(0, color=INK, lw=0.9)
    for yy, v in zip(y, vals):
        ax2.text(v + (0.3 if v >= 0 else -0.3), yy, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    lim = abs(vals).max() * 1.4
    ax2.set(ylim=(n - 0.5, -0.5), yticks=[], xlim=(-lim, lim), title="Volatility effect (%)")
    ax2.spines["left"].set_visible(False)
    for a in (ax1, ax2):
        a.grid(axis="x", color=GRID, lw=0.7)
    fig.subplots_adjust(top=0.9, bottom=0.05)
    fig.suptitle("Regime composition of the simulated scenarios", x=0.01, ha="left", fontsize=12.5, y=0.985)
    save(fig, out, "fig7_scenario_map.png")


def fig_paired_repeats(ws: WS, out: Path):
    """M=100 반복의 Base vs 시나리오 Volatility 분포 — 효과가 반복 잡음보다 큰지 보여준다."""
    raw = ws.raw["kr"]
    base = raw[raw["scenario_id"] == "Base"].sort_values("m")["Volatility"].to_numpy()
    picks = [ws.ids[0], ws.ids[1], ws.ids[len(ws.ids) // 2], ws.ids[-2], ws.ids[-1]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axhline(base.mean(), color=INK, lw=1.2, ls="--")
    ax.text(len(picks) - 0.55, base.mean(), "  Base mean", va="bottom", ha="right", fontsize=8.5, color=INK2)
    for i, sid in enumerate(picks):
        sc = raw[raw["scenario_id"] == sid].sort_values("m")["Volatility"].to_numpy()
        c, _ = _group_color(ws.scen.loc[sid])
        jitter = (np.random.default_rng(1).random(len(base)) - 0.5) * 0.22
        ax.scatter(np.full(len(base), i - 0.18) + jitter, base, s=9, color=MUTED, alpha=0.6, lw=0)
        ax.scatter(np.full(len(sc), i + 0.18) + jitter, sc, s=9, color=c, alpha=0.8, lw=0)
        ax.plot([i - 0.18, i + 0.18], [base.mean(), sc.mean()], color=INK, lw=1.2, marker="_", ms=14)
        ax.text(i, ax.get_ylim()[0] if False else min(base.min(), sc.min()),
                f"{ws.eff['kr'].loc['Volatility', sid]:+.1f}%", ha="center", va="top", fontsize=9, color=INK)
    ax.set_xticks(range(len(picks)), [f"{s}\n{ws.combo(s)}" for s in picks], fontsize=8)
    ax.set(ylabel="Daily log-return volatility of (1+GAP)(1+KP)",
           title="Baseline vs scenario across the 100 common-random-number repeats")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.legend(handles=[Patch(color=MUTED, label="Baseline repeats"),
                       Patch(color=ORANGE, label="Scenario: VKOSPI Extreme"),
                       Patch(color=AQUA, label="Scenario: Normal, KR search High"),
                       Patch(color=BLUE, label="Scenario: Normal, KR search Mid/Low")],
              loc="upper right", fontsize=8.5)
    ax.text(0.01, 0.01, "Each dot = one CRN repeat (1,000 paths); bar = repeat mean; label = effect vs baseline",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.margins(y=0.15)
    fig.tight_layout()
    save(fig, out, "fig8_paired_repeats.png")


def fig_basis_amplification(ws: WS, out: Path):
    c = ws.cmp.set_index("metric").loc[[m for m in METRICS if m in ws.cmp["metric"].values]]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(c))[::-1]
    ax.barh(y + 0.2, c["effect_median_etf"], height=0.38, color=BLUE, label="Full ETF price (etf)")
    ax.barh(y - 0.2, c["effect_median_kr"], height=0.38, color=ORANGE, label="Korean factors only (kr)")
    for yy, (e, k, a) in zip(y, zip(c["effect_median_etf"], c["effect_median_kr"], c["amplification"])):
        ax.text(max(e, k) + 0.1, yy, f"×{a:.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [METRIC_LABEL[m] for m in c.index])
    ax.set(xlabel="Median |effect| across scenarios (%)",
           title="Measurement basis: Korean-factor effects are diluted in the full price")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig9_basis_amplification.png")


# ══════════════════════════════════════════════════════════════
# 4. 실행
# ══════════════════════════════════════════════════════════════
def main(root: str, out_dir: str, use_all: bool = False):
    ws = WS(Path(root).resolve(), use_all=use_all)
    out = Path(out_dir).resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("[1] NAV 재적합 (모수 표용)")
    try:
        nav = nav_params(ws)
    except Exception as e:  # 재적합 실패해도 나머지는 만든다
        print(f"    NAV 재적합 실패: {e}")
        nav = {}

    print("[2] 표")
    T = int(ws.val["nav"]["T"])
    sample = pd.DataFrame([
        ("Sample period", f"{ws.price['Date'].iloc[0].date()} ~ {ws.price['Date'].iloc[-1].date()}"),
        ("Trading days (simulation T)", T),
        ("Baseline paths per component (N)", int(ws.meta.get("n_simulations", 1000) if "n_simulations" in ws.meta else ws.val.get("n_simulations", 1000))),
        ("CRN repeats for significance (M)", int(ws.raw["kr"]["m"].max() + 1)),
        ("Scenarios simulated", len(ws.ids)),
        ("Regime combinations observed", len(ws.table)),
        ("Theoretical combination space", int(np.prod([ws.table[a].nunique() for a in AX]))),
        ("KP observed share above τ (%)", round(float((ws.kp_sim['actual_kp'] > ws.val['kp']['threshold_ou_params']['threshold']).mean() * 100), 1)),
    ], columns=["item", "value"])
    tables = {
        "table1_sample": sample,
        "table2_parameters": table_params(ws, nav),
        "table3_validation": table_validation(ws),
        "table4_hmm_selection": table_hmm(ws),
        "table5_scenarios": table_scenarios(ws),
        "table6a_effects_kr": table_effects(ws, "kr"),
        "table6b_effects_etf": table_effects(ws, "etf"),
        "table7_basis_comparison": ws.cmp,
        "table8_base_risk": table_base_risk(ws),
        "table9_sign_summary": table_sign_summary(ws),
        "table10_group_means": table_group_means(ws),
    }
    md = [f"# Results tables — {ws.root.name}", "",
          f"Source: `{ws.root}`  |  T={T}  |  scenarios={len(ws.ids)}", ""]
    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
        md += [f"## {name}", "", md_table(df) if len(df) else "(not available)", ""]
    (out / "TABLES.md").write_text("\n".join(md), encoding="utf-8")

    print("[3] 그림")
    fig_fanchart(ws, out / "figures")
    fig_component_fit(ws, out / "figures")
    fig_validation(ws, out / "figures")
    fig_volatility_bars(ws, out / "figures")
    fig_heatmaps(ws, out / "figures")
    fig_vol_vs_dd(ws, out / "figures")
    fig_scenario_map(ws, out / "figures")
    fig_paired_repeats(ws, out / "figures")
    fig_basis_amplification(ws, out / "figures")
    print(f"완료 -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], use_all="--all" in sys.argv)
