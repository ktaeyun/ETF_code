"""
Step 3 모형 선정의 데이터 특성 근거

모형은 선행연구와 데이터 특성으로 선정한다. 이 스크립트는 그 중 '데이터 특성'
쪽을 수치로 확인한다. 각 검정은 해당 컴포넌트에 선택된 모형이 요구하는 성질을
겨냥한다.

  NAV -> ARIMA-GARCH-t : 자기상관(ARMA) + 변동성 군집(GARCH) + 두꺼운 꼬리(t)
  GAP -> OU            : 평균회귀
  KP  -> Threshold-OU  : 평균회귀 + 레짐(임계) 의존성

본 파이프라인(simulator/main.py)과 독립적이며 결과 파일만 남긴다.
실행: python analysis/data_characteristics.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from simulator.data_loader import (load_gap_exog, load_kp_exog, load_nav_exog_and_returns,
                                   align_to_nav_grid)


def _fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def autocorrelation(x, lags=10):
    """ARMA 성분 필요성: 수준(수익률)의 자기상관"""
    r = acorr_ljungbox(x, lags=[lags], return_df=True)
    return {"stat": float(r["lb_stat"].iloc[0]), "p": float(r["lb_pvalue"].iloc[0])}


def volatility_clustering(x, lags=(5, 10, 22)):
    """
    GARCH 필요성: ARCH 효과를 세 가지로 교차확인한다.
      - Engle ARCH-LM (표준 검정)
      - Ljung-Box on r^2
      - Ljung-Box on |r|  (제곱보다 이상치에 덜 휘둘려 더 민감한 경우가 많다)
    래그 선택에 결과가 좌우되지 않는지 보려고 여러 래그를 함께 낸다.
    """
    x = np.asarray(x, dtype=float)
    out = {}
    for L in lags:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm_p = float(het_arch(x, nlags=L)[1])
        out[L] = {
            "arch_lm_p": lm_p,
            "lb_sq_p": float(acorr_ljungbox(x ** 2, lags=[L], return_df=True)["lb_pvalue"].iloc[0]),
            "lb_abs_p": float(acorr_ljungbox(np.abs(x), lags=[L], return_df=True)["lb_pvalue"].iloc[0]),
        }
    return out


def fat_tails(x):
    """Student-t 필요성: 초과첨도와 정규성 기각"""
    x = np.asarray(x, dtype=float)
    jb, jb_p = stats.jarque_bera(x)
    return {
        "excess_kurtosis": float(stats.kurtosis(x)),
        "skew": float(stats.skew(x)),
        "jb": float(jb),
        "jb_p": float(jb_p),
    }


def mean_reversion(x):
    """OU 필요성: 단위근 기각(ADF) + 정상성 유지(KPSS). 둘이 같은 방향이어야 강한 근거."""
    x = np.asarray(x, dtype=float)
    adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto")[:2]

    # ADF는 래그 선택에 민감할 수 있어 고정 래그도 함께 본다.
    # 사양에 따라 결론이 갈리면 그 자체가 '근거가 약하다'는 신호다.
    alt = {}
    for lag in (1, 4):
        alt[f"adf_lag{lag}_p"] = float(adfuller(x, maxlag=lag, autolag=None)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alt["kpss_ct_p"] = float(kpss(x, regression="ct", nlags="auto")[1])

    # OU 반감기: 중심화 후 AR(1) 적합 -> kappa = -log(phi), 반감기 = log2/kappa
    d = x - x.mean()
    phi = float(np.dot(d[:-1], d[1:]) / np.dot(d[:-1], d[:-1]))
    half_life = float(np.log(2) / -np.log(phi)) if 0 < phi < 1 else float("nan")
    res = {
        "adf_stat": float(adf_stat), "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p),
        "ar1_phi": phi, "half_life_days": half_life,
    }
    res.update(alt)
    return res


def _phi_gap(series, n_grid=50):
    """중심화 계열에서 |x|<=tau 와 |x|>tau 의 AR(1) phi 격차가 최대인 tau를 찾는다."""
    d = np.asarray(series, dtype=float)
    d = d - d.mean()
    best = (np.nan, np.nan, np.nan, np.nan)
    for tau in np.quantile(np.abs(d), np.linspace(0.3, 0.8, n_grid)):
        inner = np.abs(d) <= tau
        a1, b1 = d[:-1][inner[:-1]], d[1:][inner[:-1]]
        a2, b2 = d[:-1][~inner[:-1]], d[1:][~inner[:-1]]
        if len(a1) < 20 or len(a2) < 20:
            continue
        if np.dot(a1, a1) == 0 or np.dot(a2, a2) == 0:
            continue
        p1 = np.dot(a1, b1) / np.dot(a1, a1)
        p2 = np.dot(a2, b2) / np.dot(a2, a2)
        if not (np.isfinite(p1) and np.isfinite(p2)):
            continue
        if not np.isfinite(best[0]) or abs(p1 - p2) > best[0]:
            best = (abs(p1 - p2), float(tau), float(p1), float(p2))
    return best


def threshold_effect(x, n_perm=500, block=20, seed=42):
    """
    Threshold-OU 필요성: 레짐별 회귀속도가 다른가.
    귀무(임계구조 없음) 분포는 블록 순열로 근사한다 — 블록을 섞으면 임계 구조는
    깨지지만 블록 내부의 자기상관은 대체로 보존된다.
    """
    x = np.asarray(x, dtype=float)
    gap, tau, phi_in, phi_out = _phi_gap(x)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(x) / block))
    null = []
    for _ in range(n_perm):
        order = rng.permutation(n_blocks)
        perm = np.concatenate([x[i * block:(i + 1) * block] for i in order])[:len(x)]
        g = _phi_gap(perm)[0]
        if np.isfinite(g):
            null.append(g)
    null = np.asarray(null)
    p = float(np.mean(null >= gap)) if null.size and np.isfinite(gap) else float("nan")
    return {"tau": tau, "phi_inner": phi_in, "phi_outer": phi_out,
            "phi_gap": gap, "perm_p": p, "n_null": int(null.size)}


def main():
    # NAV 그리드(결정 11: BTC 현물 + 1거래일 시차보정)에 GAP/KP를 맞춘다.
    _nav_df = load_nav_exog_and_returns()
    _gap_df, _kp_df = align_to_nav_grid(_nav_df, load_gap_exog(), load_kp_exog())
    nav_ret = _nav_df["Log Return"].to_numpy(dtype=float)
    nav_ret = nav_ret[np.isfinite(nav_ret)]
    gap = _gap_df["etf_premium"].to_numpy(dtype=float)
    gap = gap[np.isfinite(gap)]
    kp = _kp_df["Kimchi Premium"].to_numpy(dtype=float)
    kp = kp[np.isfinite(kp)]

    print("=" * 78)
    print("Step 3 모형 선정 - 데이터 특성 근거")
    print("=" * 78)
    rows = []

    # ---- NAV -> ARIMA-GARCH-t ----
    print(f"\n[NAV] Log Return, T={len(nav_ret)}  ->  ARIMA-GARCH-t")
    ac, vc, ft = autocorrelation(nav_ret), volatility_clustering(nav_ret), fat_tails(nav_ret)
    print(f"  자기상관    Ljung-Box(10)        Q={ac['stat']:8.2f}  p={_fmt_p(ac['p'])}"
          f"   -> ARMA {'필요' if ac['p'] < 0.05 else '근거 약함'}")
    print("  변동성군집  (ARCH 효과 — 세 검정 x 세 래그로 교차확인)")
    for L, d in vc.items():
        print(f"              lag={L:2d}   ARCH-LM p={d['arch_lm_p']:.3f}   "
              f"LB(r^2) p={d['lb_sq_p']:.3f}   LB(|r|) p={d['lb_abs_p']:.3f}")
    vc_any = any(min(d.values()) < 0.05 for d in vc.values())
    print(f"              -> GARCH {'필요' if vc_any else '근거 약함 (어느 조합에서도 기각 안 됨)'}")
    print(f"  두꺼운꼬리  초과첨도={ft['excess_kurtosis']:+.3f} 왜도={ft['skew']:+.3f} "
          f"JB p={_fmt_p(ft['jb_p'])}   -> Student-t {'필요' if ft['jb_p'] < 0.05 else '근거 약함'}")
    rows += [("NAV", "자기상관 (Ljung-Box 10)", ac["stat"], ac["p"]),
             *[(f"NAV", f"ARCH 효과 lag={L} ({k})", float("nan"), d[k])
               for L, d in vc.items() for k in ("arch_lm_p", "lb_sq_p", "lb_abs_p")],
             ("NAV", "정규성 (Jarque-Bera)", ft["jb"], ft["jb_p"])]

    # ---- GAP -> OU ----
    print(f"\n[GAP] etf_premium, T={len(gap)}  ->  OU")
    mr, ftg = mean_reversion(gap), fat_tails(np.diff(gap))
    print(f"  평균회귀    ADF ={mr['adf_stat']:8.3f}  p={_fmt_p(mr['adf_p'])}  (단위근 기각 필요)")
    print(f"              KPSS={mr['kpss_stat']:8.3f}  p={_fmt_p(mr['kpss_p'])}  (정상성 유지 필요)")
    print(f"              AR(1) phi={mr['ar1_phi']:.4f}  반감기={mr['half_life_days']:.1f}일"
          f"   -> OU {'적합' if mr['adf_p'] < 0.05 else '근거 약함'}")
    print(f"  (참고) 변화량 초과첨도={ftg['excess_kurtosis']:+.3f}  JB p={_fmt_p(ftg['jb_p'])}"
          f"   -> Gaussian 혁신항의 한계")
    rows += [("GAP", "단위근 (ADF)", mr["adf_stat"], mr["adf_p"]),
             ("GAP", "정상성 (KPSS)", mr["kpss_stat"], mr["kpss_p"]),
             ("GAP", "변화량 정규성 (JB)", ftg["jb"], ftg["jb_p"])]

    # ---- KP -> Threshold-OU ----
    print(f"\n[KP] Kimchi Premium, T={len(kp)}  ->  Threshold-OU")
    mrk, th = mean_reversion(kp), threshold_effect(kp)
    print(f"  평균회귀    ADF(AIC)={mrk['adf_stat']:7.3f}  p={_fmt_p(mrk['adf_p'])}"
          f"   AR(1) phi={mrk['ar1_phi']:.4f}  반감기={mrk['half_life_days']:.1f}일")
    print(f"              사양 민감도: ADF(lag1) p={mrk['adf_lag1_p']:.3f}  "
          f"ADF(lag4) p={mrk['adf_lag4_p']:.3f}  KPSS(ct) p={mrk['kpss_ct_p']:.3f}")
    _ps = [mrk['adf_p'], mrk['adf_lag1_p'], mrk['adf_lag4_p']]
    print(f"              -> 평균회귀 "
          f"{'일관되게 지지' if max(_ps) < 0.05 else '사양에 민감 (근거 약함)'}"
          f"{'; KPSS가 정상성 기각' if mrk['kpss_p'] < 0.05 else ''}")
    print(f"  임계효과    tau={th['tau']:.4f}  phi(내부)={th['phi_inner']:.4f}  "
          f"phi(외부)={th['phi_outer']:.4f}")
    print(f"              격차={th['phi_gap']:.4f}  블록순열 p={_fmt_p(th['perm_p'])} "
          f"(B={th['n_null']})   -> Threshold {'필요' if th['perm_p'] < 0.05 else '근거 약함'}")
    rows += [("KP", "단위근 (ADF)", mrk["adf_stat"], mrk["adf_p"]),
             ("KP", "임계효과 (블록순열)", th["phi_gap"], th["perm_p"])]

    out_dir = _root / "results" / "data_characteristics"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["component", "test", "statistic", "pvalue"])
    df["reject_at_05"] = df["pvalue"] < 0.05
    path = out_dir / "data_characteristics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
