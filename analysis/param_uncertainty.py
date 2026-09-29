# -*- coding: utf-8 -*-
"""
모수 불확실성 강건성: 부트스트랩으로 뽑은 δ 계수 세트마다 축별 시나리오 쌍의 효과 차이를
다시 계산해, 방향(부호)이 유지되는 비율을 낸다.

    python analysis/param_uncertainty.py <작업공간> <출력폴더> [--B 200] [--N 200]

왜 필요한가
    유의성 검정(M=100 CRN)은 시뮬레이션 잡음만 검정하고 모수 불확실성은 다루지 않는다.
    KP·GAP 변동성 식의 δ 계수는 부트스트랩 95% CI가 모두 0을 포함할 만큼 약하게 식별되므로
    (결정 12), "쌍 11/11 일치" 같은 결과가 계수 부호에 얼마나 의존하는지 보여야 한다.

방법
    1. Base 적합과 같은 방식으로 δ를 재추정하되, 잔차 회귀 표본을 부트스트랩해 B개 세트를 만든다
       (KP는 레짐별 log|잔차| 회귀, GAP은 log|잔차| ~ RV 회귀). κ·μ·σ0·τ는 고정한다.
    2. 각 세트로 축별 대표 쌍(다른 네 축이 같은 두 시나리오)을 N개 경로씩 시뮬레이션해
       한국 요인 변동성 효과 차이(높은 수준 − 낮은 수준)를 계산한다.
    3. 축별로 부호 유지 비율과 차이의 분포(2.5/50/97.5 분위)를 보고한다.

출력
    tables/param_uncertainty_<metric>.csv, figures/fig11_param_uncertainty.png
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AX = ["Bitcoin_RV", "VKOSPI", "KR_Volume", "KR_SVI", "Global_SVI"]
ORDER = {"Low": 0, "Mid": 1, "High": 2, "Normal": 0, "Extreme": 1}
# KP σ_t 의 어느 δ 가 어느 축에 대응하는가 (GAP 은 별도)
KP_AXIS = {"KR_Volume": "delta1", "VKOSPI": "delta2", "KR_SVI": "delta3"}
GAP_AXIS = {"Global_SVI": "delta1", "Bitcoin_RV": "delta2"}
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
AX_COLOR = dict(zip(AX, [VIOLET, ORANGE, BLUE, AQUA, YELLOW]))


def zc(x, clip=3.0):
    x = np.asarray(x, dtype=float)
    return np.clip((x - x.mean()) / (x.std() + 1e-8), -clip, clip)


def bootstrap_deltas(ws: Path, meta: dict, B: int, seed: int = 0):
    """Base 적합과 같은 회귀를 부트스트랩해 δ 세트 B개를 만든다 (κ·μ·σ0·τ 고정)."""
    sys.path.insert(0, str(ws))
    from simulator.data_loader import load_gap_exog, load_kp_exog, load_nav_exog_and_returns, align_to_nav_grid
    with contextlib.redirect_stdout(io.StringIO()):
        nav = load_nav_exog_and_returns(base_dir=str(ws), nav_source=meta.get("nav_source", "btc"))
        dg, dk = align_to_nav_grid(nav, load_gap_exog(base_dir=str(ws)), load_kp_exog(base_dir=str(ws)))
    rng = np.random.default_rng(seed)

    # ── KP: 레짐별 log|잔차| ~ [vol, vkospi, svi] ──────────────────
    kp = dk["Kimchi Premium"].to_numpy(dtype=float)
    kpp = meta["kp_params"]
    tau = kpp["threshold"]
    zv = zc(np.log(dk["volume_btc"].to_numpy(dtype=float) + 1e-8))
    zk = zc(dk["KOSPI_Volatility"].to_numpy(dtype=float))
    zs = zc(dk["bitcoin_kr"].to_numpy(dtype=float))
    d_kp, lag = np.diff(kp), kp[:-1]
    kp_sets = {}
    for r, p in kpp["regime_params"].items():
        mask = (lag <= tau) if int(r) == 0 else (lag > tau)
        resid = d_kp[mask] - p["kappa"] * (p["mu"] - lag[mask])
        y = np.log(np.abs(resid) + 1e-8)
        X = np.column_stack([np.ones(mask.sum()), zv[:-1][mask], zk[:-1][mask], zs[:-1][mask]])
        n = mask.sum()
        draws = []
        for _ in range(B):
            i = rng.integers(0, n, n)
            b = np.linalg.lstsq(X[i], y[i], rcond=None)[0] / 2.0
            draws.append(np.clip(b[1:], -2.0, 2.0))
        kp_sets[int(r)] = np.array(draws)  # (B, 3) = delta1, delta2, delta3

    # ── GAP: log|잔차| ~ [SI, RV] ────────────────────────────────
    gap = dg["etf_premium"].to_numpy(dtype=float)
    gp = meta["gap_params"]
    zsi = zc(dg["value"].to_numpy(dtype=float))
    zrv = zc(dg["btc_volatility"].to_numpy(dtype=float))
    d_g, glag = np.diff(gap), gap[:-1]
    resid = d_g - gp["kappa"] * (gp["mu"] - glag)
    y = np.log(np.abs(resid) + 1e-8)
    X = np.column_stack([np.ones(len(y)), zsi[:-1], zrv[:-1]])
    n = len(y)
    gap_sets = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        b = np.linalg.lstsq(X[i], y[i], rcond=None)[0] / 2.0
        gap_sets.append(np.clip(b[1:], -2.0, 2.0))
    return np.array(gap_sets), kp_sets, dg, dk


def pick_pairs(ws: Path, eff_ids):
    """축별 대표 쌍: 다른 네 축이 같고 관측일수 합이 가장 큰 쌍."""
    scen = pd.read_csv(ws / "results" / "scenario_selection" / "final_scenarios_latest.csv").set_index("Scenario_ID")
    scen = scen.loc[[i for i in eff_ids if i in scen.index]]
    best = {}
    import itertools
    for a, b in itertools.combinations(scen.index, 2):
        diff = [x for x in AX if scen.loc[a, x] != scen.loc[b, x]]
        if len(diff) != 1:
            continue
        x = diff[0]
        lo, hi = (a, b) if ORDER[scen.loc[a, x]] < ORDER[scen.loc[b, x]] else (b, a)
        w = scen.loc[lo, "joint_frequency"] + scen.loc[hi, "joint_frequency"]
        if x not in best or w > best[x][2]:
            best[x] = (lo, hi, w)
    return scen, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("out")
    ap.add_argument("--B", type=int, default=200, help="부트스트랩 δ 세트 수")
    ap.add_argument("--N", type=int, default=200, help="세트당 경로 수")
    a = ap.parse_args()
    ws, out = Path(a.workspace).resolve(), Path(a.out).resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    meta = json.loads((ws / "results" / "simulator" / "cache" / "sim_meta.json").read_text(encoding="utf-8"))
    eff = pd.read_csv(ws / "results" / "scenario_simulator" / "significance_test" /
                      "effect_size_matrix_pct_kr.csv", index_col=0)
    scen, pairs = pick_pairs(ws, list(eff.columns))
    print(f"대표 쌍: " + ", ".join(f"{x}: {lo}->{hi} (n={w})" for x, (lo, hi, w) in pairs.items()))

    gap_sets, kp_sets, dg, dk = bootstrap_deltas(ws, meta, a.B)
    from simulator.gap_ou_simulator import GapOUSimulator
    from simulator.kp_threshold_ou_simulator import KPThresholdOUSimulator
    from simulator.scenario_main import _build_hmm_with_fixed_k
    from preprocessing.scenario_generator import generate_scenario_series
    with contextlib.redirect_stdout(io.StringIO()):
        hmm = _build_hmm_with_fixed_k(ws / "results" / "cache" / "regime_df_cache.csv")

    T = min(len(dg), len(dk))
    VAR = {"Bitcoin_RV": "Global_RV", "VKOSPI": "VKOSPI_resid", "KR_Volume": "btc_volume_btc",
           "KR_SVI": "domestic_btc_svi", "Global_SVI": "global_btc_svi"}
    LEVEL = {"Low": 0, "Mid": 1, "High": 2, "Normal": 0, "Extreme": 1}

    def exog_for(sid, seed=42):
        sc = {VAR[x]: LEVEL[str(scen.loc[sid, x])] for x in AX}
        df = generate_scenario_series(hmm, sc, T=T, seed=seed, inverse=True)
        return dict(si=df["global_btc_svi"].to_numpy()[:T], vix=df["Global_RV"].to_numpy()[:T],
                    vol=df["btc_volume_btc"].to_numpy()[:T], kv=df["VKOSPI_resid"].to_numpy()[:T],
                    bkr=df["domestic_btc_svi"].to_numpy()[:T])

    gp, kpp = meta["gap_params"], meta["kp_params"]
    log_vol = np.log(dk["volume_btc"].to_numpy(dtype=float) + 1e-8)
    stats = dict(si_mean=float(dg["value"].mean()), si_std=float(dg["value"].std()),
                 vix_mean=float(dg["btc_volatility"].mean()), vix_std=float(dg["btc_volatility"].std()),
                 vol_mean=float(log_vol.mean()), vol_std=float(log_vol.std()),
                 kospi_mean=float(dk["KOSPI_Volatility"].mean()), kospi_std=float(dk["KOSPI_Volatility"].std()),
                 bkr_mean=float(dk["bitcoin_kr"].mean()), bkr_std=float(dk["bitcoin_kr"].std()))

    def run(exog, g_delta, kp_delta, n, seed0):
        gap_sim = GapOUSimulator(kappa=gp["kappa"], mu=gp["mu"], sigma0=gp["sigma0"],
                                 delta1=g_delta[0], delta2=g_delta[1],
                                 si_mean=stats["si_mean"], si_std=stats["si_std"],
                                 vix_mean=stats["vix_mean"], vix_std=stats["vix_std"],
                                 clip=3.0, nu=gp.get("nu"))
        rp = kpp["regime_params"]
        kp_sim = KPThresholdOUSimulator({
            "threshold": kpp["threshold"], "nu": kpp.get("nu"),
            "n_regimes": int(kpp.get("n_regimes") or len(rp)),
            "regime_params": {int(r): {"kappa": v["kappa"], "mu": v["mu"], "sigma0": v["sigma0"]} for r, v in rp.items()},
            "delta1_regime": {r: kp_delta[r][0] for r in kp_delta},
            "delta2_regime": {r: kp_delta[r][1] for r in kp_delta},
            "delta3_regime": {r: kp_delta[r][2] for r in kp_delta},
            "vol_btc_mean": stats["vol_mean"], "vol_btc_std": stats["vol_std"],
            "kospi_mean": stats["kospi_mean"], "kospi_std": stats["kospi_std"],
            "bitcoin_kr_mean": stats["bkr_mean"], "bitcoin_kr_std": stats["bkr_std"],
        }, clip=3.0)
        g0, k0 = float(dg["etf_premium"].iloc[0]), float(dk["Kimchi Premium"].iloc[0])
        mg = np.array([np.asarray(gap_sim.simulate_gap(T=T, g0=g0, si_future=exog["si"], vix_future=exog["vix"],
                                                       seed=seed0 + 10_000 + i)).ravel() for i in range(n)])
        mk = np.array([np.asarray(kp_sim.simulate_kp(T=T, kp0=k0, volume_btc_future=exog["vol"],
                                                     kospi_vol_future=exog["kv"], bitcoin_kr_future=exog["bkr"],
                                                     seed=seed0 + 20_000 + i)).ravel() for i in range(n)])
        path = (1 + mg) * (1 + mk)
        lr = np.diff(np.log(np.maximum(path, 1e-12)), axis=1)
        return float(lr.std(axis=1).mean())

    rows = []
    for x, (lo, hi, w) in pairs.items():
        e_lo, e_hi = exog_for(lo), exog_for(hi)
        diffs = []
        for b in range(a.B):
            gd = gap_sets[b]
            kd = {r: kp_sets[r][b] for r in kp_sets}
            v_lo = run(e_lo, gd, kd, a.N, 1000)
            v_hi = run(e_hi, gd, kd, a.N, 1000)
            diffs.append((v_hi / v_lo - 1.0) * 100)
        d = np.array(diffs)
        rows.append({"axis": x, "pair": f"{lo}->{hi}", "n_days": int(w),
                     "median_pct": np.median(d), "q2.5": np.percentile(d, 2.5), "q97.5": np.percentile(d, 97.5),
                     "share_positive": float((d > 0).mean()),
                     "direction_stable": float(max((d > 0).mean(), (d < 0).mean()))})
        print(f"  {x:11s} {lo}->{hi}: 중앙 {np.median(d):+.2f}%  [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]  "
              f"양(+) 비율 {(d>0).mean()*100:.0f}%")
        np.save(out / "tables" / f"_pu_draws_{x}.npy", d)

    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "param_uncertainty_Volatility.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, x in enumerate(AX):
        f = out / "tables" / f"_pu_draws_{x}.npy"
        if not f.exists():
            ax.text(i, 0, "no pair", ha="center", fontsize=8.5, color=MUTED)
            continue
        d = np.load(f)
        j = (np.random.default_rng(1).random(len(d)) - 0.5) * 0.3
        ax.scatter(np.full(len(d), i) + j, d, s=10, color=AX_COLOR[x], alpha=0.5, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color=INK, lw=1.6, zorder=4)
        r = df.loc[df.axis == x].iloc[0]
        ax.text(i, d.max() + 0.4, f"{r['share_positive']*100:.0f}% +", ha="center", fontsize=9, color=INK2)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(range(len(AX)), AX, fontsize=9)
    ax.set(ylabel="Volatility effect of the axis, higher − lower (%)",
           title="Parameter uncertainty: axis direction across bootstrap δ draws")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.text(0.01, 0.01, f"B={a.B} bootstrap δ sets × N={a.N} paths; κ, μ, σ₀, τ held at the base fit",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig11_param_uncertainty.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n저장: {out/'tables'/'param_uncertainty_Volatility.csv'}, {out/'figures'/'fig11_param_uncertainty.png'}")


if __name__ == "__main__":
    main()
