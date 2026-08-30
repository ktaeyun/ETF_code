"""
시나리오별 GAP / KP / ETF 분포 형태 진단 (discussion 자료)

배경 — Step 11 유의성 검정에서 비대칭이 관찰됐다.
  VaR_99 (하방 1% 꼬리) : 9개 시나리오 모두 유의, 효과크기 0.80%
  p95    (상방 5% 꼬리) : 대부분 유의하지 않고 효과크기 0.13%
같은 시나리오 조작이 아래쪽 꼬리는 움직이는데 위쪽 꼬리는 안 움직인다.

이 비대칭이 GAP·KP 자체의 분포 비대칭(왜도)에서 오는지 확인한다.

산출:
  1. 시나리오별 GAP, KP 의 왜도와 초과첨도
  2. 시나리오별 최종 시점 ETF 가격의 왜도 (전체 ETF / 한국 요인 두 기준)
  3. S05(위기) 대 S01·S03(평온)에서 하방(1%, 5%)과 상방(95%) 꼬리의 이동폭

본 파이프라인과 독립이며 results/discussion/ 아래에만 결과를 남긴다.
실행: python analysis/scenario_distribution_shape.py
"""
from __future__ import annotations

import contextlib
import io as _io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib
matplotlib.use("Agg")

from preprocessing.scenario_generator import generate_scenario_series
from simulator.data_loader import load_gap_exog, load_kp_exog
from simulator.gap_ou_simulator import GapOUSimulator
from simulator.kp_threshold_ou_simulator import KPThresholdOUSimulator
from simulator.scenario_main import _build_hmm_with_fixed_k

ANCHOR = 10000.0
N_PATHS = 1000
SEED = 42


def _build_simulators():
    """Base 적합 모수로 GAP / KP 시뮬레이터를 만든다 (시나리오 간 모수 고정)."""
    meta = json.loads((_root / "results" / "simulator" / "cache" / "sim_meta.json")
                      .read_text(encoding="utf-8"))
    gp, kpp = meta["gap_params"], meta["kp_params"]
    dg, dk = load_gap_exog(), load_kp_exog()

    gap_sim = GapOUSimulator(
        kappa=gp["kappa"], mu=gp["mu"], sigma0=gp["sigma0"],
        delta1=gp["delta1"], delta2=gp["delta2"],
        si_mean=float(dg["value"].mean()), si_std=float(dg["value"].std()),
        vix_mean=float(dg["btc_volatility"].mean()),
        vix_std=float(dg["btc_volatility"].std()),
        clip=3.0, nu=gp.get("nu"),
    )
    rp = kpp["regime_params"]
    log_vol = np.log(dk["volume_btc"].values + 1e-8)
    kp_sim = KPThresholdOUSimulator({
        "threshold": kpp["threshold"],
        "n_regimes": int(kpp.get("n_regimes") or len(rp)),
        "regime_params": {int(r): {"kappa": v["kappa"], "mu": v["mu"],
                                   "sigma0": v["sigma0"]} for r, v in rp.items()},
        "delta1_regime": {int(r): v["delta1"] for r, v in rp.items()},
        "delta2_regime": {int(r): v["delta2"] for r, v in rp.items()},
        "delta3_regime": {int(r): v["delta3"] for r, v in rp.items()},
        "vol_btc_mean": float(log_vol.mean()), "vol_btc_std": float(log_vol.std()),
        "kospi_mean": float(dk["KOSPI_Volatility"].mean()),
        "kospi_std": float(dk["KOSPI_Volatility"].std()),
        "bitcoin_kr_mean": float(dk["bitcoin_kr"].mean()),
        "bitcoin_kr_std": float(dk["bitcoin_kr"].std()),
    }, clip=3.0)
    return gap_sim, kp_sim, dg, dk


def _regime_index_map(hmm):
    """레짐 라벨 -> 인덱스. 변수마다 K가 달라(2 또는 3) 따로 매핑한다."""
    K = {k: len(v.mu) for k, v in hmm.items()}

    def idx(var, label):
        if K[var] == 2:
            return {"Low": 0, "High": 1, "Normal": 0, "Extreme": 1, "Mid": 1}[label]
        return {"Low": 0, "Mid": 1, "High": 2, "Normal": 0, "Extreme": 1}[label]
    return idx


def simulate(gap_sim, kp_sim, dg, dk, exog):
    """주어진 외생변수로 GAP / KP 경로 N개씩 생성."""
    T = len(dg)
    mc_gap = np.array([
        np.asarray(gap_sim.simulate_gap(
            T=T, g0=float(dg["etf_premium"].iloc[0]),
            si_future=exog["si"], vix_future=exog["vix"],
            seed=SEED + 10_000 + i)).flatten()
        for i in range(N_PATHS)
    ])
    mc_kp = np.array([
        np.asarray(kp_sim.simulate_kp(
            T=T, kp0=float(dk["Kimchi Premium"].iloc[0]),
            volume_btc_future=exog["vol"], kospi_vol_future=exog["kv"],
            bitcoin_kr_future=exog["bkr"], seed=SEED + 20_000 + i)).flatten()
        for i in range(N_PATHS)
    ])
    return mc_gap, mc_kp


def anchored(raw):
    init = raw[:, 0:1]
    init = np.where(init == 0, 1.0, init)
    return raw * (ANCHOR / init)


def main():
    warnings.filterwarnings("ignore")
    out_dir = _root / "results" / "discussion"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    gap_sim, kp_sim, dg, dk = _build_simulators()
    with contextlib.redirect_stdout(_io.StringIO()):
        hmm = _build_hmm_with_fixed_k(_root / "results" / "cache" / "regime_df_cache.csv")
    idx = _regime_index_map(hmm)
    scen = pd.read_csv(_root / "results" / "scenario_selection" /
                       "final_scenarios_latest.csv")
    T = len(dg)
    mc_nav = np.load(_root / "results" / "simulator" / "cache" /
                     "sim_arrays.npz")["mc_nav"][:, :T]

    P("=" * 90)
    P("시나리오별 GAP / KP / ETF 분포 형태")
    P("=" * 90)
    P(f"경로 {N_PATHS}개 x T={T}, seed={SEED}, 모수는 Base 고정 (외생변수만 시나리오별 교체)")

    targets = [("Base", None)] + [(r["Scenario_ID"], r) for _, r in scen.iterrows()]
    store, rows = {}, []

    for sid, row in targets:
        if row is None:
            exog = {"si": dg["value"].values, "vix": dg["btc_volatility"].values,
                    "vol": dk["volume_btc"].values, "kv": dk["KOSPI_Volatility"].values,
                    "bkr": dk["bitcoin_kr"].values}
        else:
            s = {"Global_RV": idx("Global_RV", row["Bitcoin_RV"]),
                 "VKOSPI_resid": idx("VKOSPI_resid", row["VKOSPI"]),
                 "btc_volume_btc": idx("btc_volume_btc", row["KR_Volume"]),
                 "domestic_btc_svi": idx("domestic_btc_svi", row["KR_SVI"]),
                 "global_btc_svi": idx("global_btc_svi", row["Global_SVI"])}
            gen = generate_scenario_series(hmm, s, T=T, seed=SEED, inverse=True)
            exog = {"si": gen["global_btc_svi"].values, "vix": gen["Global_RV"].values,
                    "vol": gen["btc_volume_btc"].values, "kv": gen["VKOSPI_resid"].values,
                    "bkr": gen["domestic_btc_svi"].values}

        g, k = simulate(gap_sim, kp_sim, dg, dk, exog)
        etf = anchored(mc_nav * (1 + g) * (1 + k))
        kr = anchored((1 + g) * (1 + k))
        store[sid] = {"gap": g, "kp": k, "etf": etf, "kr": kr}

        # 1) GAP / KP 분포 형태: 전체 관측을 풀어서 (경로 x 시점)
        rows.append({
            "scenario_id": sid,
            "gap_skew": float(stats.skew(g.ravel())),
            "gap_exkurt": float(stats.kurtosis(g.ravel())),
            "kp_skew": float(stats.skew(k.ravel())),
            "kp_exkurt": float(stats.kurtosis(k.ravel())),
            # 2) 최종 시점 가격의 왜도
            "etf_terminal_skew": float(stats.skew(etf[:, -1])),
            "etf_terminal_exkurt": float(stats.kurtosis(etf[:, -1])),
            "kr_terminal_skew": float(stats.skew(kr[:, -1])),
            "kr_terminal_exkurt": float(stats.kurtosis(kr[:, -1])),
            # 꼬리 분위수 (전체 ETF 기준)
            "etf_p1": float(np.percentile(etf[:, -1], 1)),
            "etf_p5": float(np.percentile(etf[:, -1], 5)),
            "etf_p50": float(np.percentile(etf[:, -1], 50)),
            "etf_p95": float(np.percentile(etf[:, -1], 95)),
            "etf_p99": float(np.percentile(etf[:, -1], 99)),
            # 꼬리 분위수 (한국 요인 기준)
            "kr_p1": float(np.percentile(kr[:, -1], 1)),
            "kr_p5": float(np.percentile(kr[:, -1], 5)),
            "kr_p50": float(np.percentile(kr[:, -1], 50)),
            "kr_p95": float(np.percentile(kr[:, -1], 95)),
        })
        print(f"  [{sid}] 완료")

    df = pd.DataFrame(rows).set_index("scenario_id")

    # ── 1. GAP / KP 분포 형태 ─────────────────────────────────
    P("\n" + "-" * 90)
    P("1. GAP / KP 시뮬레이션 경로의 분포 형태")
    P("-" * 90)
    P(f"  {'ID':6}{'GAP 왜도':>11}{'GAP 초과첨도':>14}{'KP 왜도':>11}{'KP 초과첨도':>14}")
    for sid in df.index:
        r = df.loc[sid]
        P(f"  {sid:6}{r['gap_skew']:+11.3f}{r['gap_exkurt']:+14.3f}"
          f"{r['kp_skew']:+11.3f}{r['kp_exkurt']:+14.3f}")

    # ── 2. 최종 ETF 가격의 왜도 ───────────────────────────────
    P("\n" + "-" * 90)
    P("2. 최종 시점 가격의 왜도 / 초과첨도")
    P("-" * 90)
    P(f"  {'ID':6}{'ETF 왜도':>11}{'ETF 초과첨도':>14}{'한국요인 왜도':>15}{'한국요인 초과첨도':>18}")
    for sid in df.index:
        r = df.loc[sid]
        P(f"  {sid:6}{r['etf_terminal_skew']:+11.3f}{r['etf_terminal_exkurt']:+14.3f}"
          f"{r['kr_terminal_skew']:+15.3f}{r['kr_terminal_exkurt']:+18.3f}")

    # ── 3. 꼬리 이동 ──────────────────────────────────────────
    P("\n" + "-" * 90)
    P("3. 위기(S05) 대 평온(S01, S03): 꼬리 이동폭 (Base 대비 %)")
    P("-" * 90)
    for basis, pre, cols in [("전체 ETF", "etf", ["p1", "p5", "p50", "p95"]),
                             ("한국 요인", "kr", ["p1", "p5", "p50", "p95"])]:
        P(f"\n  [{basis} 기준]")
        P(f"  {'ID':6}" + "".join(f"{c:>12}" for c in
                                  ["하위1%", "하위5%", "중앙값", "상위95%"]))
        b = df.loc["Base"]
        for sid in ["S01", "S03", "S05"]:
            r = df.loc[sid]
            vals = [(r[f"{pre}_{c}"] - b[f"{pre}_{c}"]) / b[f"{pre}_{c}"] * 100 for c in cols]
            P(f"  {sid:6}" + "".join(f"{v:+11.3f}%" for v in vals))

    P("\n  하방/상방 이동 비대칭 (|하위1% 이동| / |상위95% 이동|)")
    for basis, pre in [("전체 ETF", "etf"), ("한국 요인", "kr")]:
        b = df.loc["Base"]
        P(f"    [{basis}]")
        for sid in ["S01", "S03", "S05"]:
            r = df.loc[sid]
            dn = abs((r[f"{pre}_p1"] - b[f"{pre}_p1"]) / b[f"{pre}_p1"] * 100)
            up = abs((r[f"{pre}_p95"] - b[f"{pre}_p95"]) / b[f"{pre}_p95"] * 100)
            ratio = dn / up if up > 1e-9 else float("nan")
            P(f"      {sid}: 하방 {dn:.3f}% / 상방 {up:.3f}% = {ratio:.2f}배")

    df.to_csv(out_dir / "scenario_distribution_shape.csv", encoding="utf-8-sig")
    (out_dir / "scenario_distribution_shape.txt").write_text("\n".join(lines),
                                                             encoding="utf-8")
    P(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
