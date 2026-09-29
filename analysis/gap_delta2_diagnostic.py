"""
GAP 모형 delta2 진단 (discussion 자료)

sigma_t = sigma_0 * exp(delta1*SI + delta2*RV) 에서 delta2 < 0 으로 추정된다.
"BTC 실현변동성이 높을수록 괴리율 변동성이 작아진다"는 뜻이라 직관과 반대다.
이 스크립트는 그 추정치가 얼마나 믿을 만한지 다섯 가지로 확인한다.

  1. 각 모수의 표준오차 / t값 / 95% 신뢰구간   (Hessian 기반 + 모수 부트스트랩)
  2. delta2 의 통계적 유의성
  3. SI 와 RV 의 상관 — 공선성으로 계수가 상쇄됐을 가능성
  4. delta2 = 0 제약 대 자유 추정의 로그우도 비교 (LR 검정)
  5. 표본 전반/후반에서 delta2 부호가 유지되는지

주의 — 두 종류의 추정치를 구분한다.
  · 운영 추정치 : 파이프라인이 실제로 쓰는 값. 목적함수에 능형 벌점
                  0.01*(delta1^2 + delta2^2) 이 붙어 있어 0 쪽으로 수축돼 있다.
  · MLE        : 벌점을 뺀 순수 최대우도 추정치. 표준오차·LR 검정은 이쪽 기준이라야
                  통계적 해석이 성립한다.

본 파이프라인과 독립이며 results/discussion/ 아래에만 결과를 남긴다.
실행: python analysis/gap_delta2_diagnostic.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import t as student_t

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from simulator.data_loader import load_gap_exog, load_nav_exog_and_returns, align_to_nav_grid
from simulator.gap_ou_simulator import (
    NU_MAX,
    NU_MIN,
    _t_scale,
    fit_ou_with_exog,
    standardized_t,
    zscore_clip,
)

PARAM_NAMES = ["kappa", "mu", "sigma0", "delta1", "delta2", "nu"]


# ──────────────────────────────────────────────────────────────
# 우도
# ──────────────────────────────────────────────────────────────
def build_design(gap, si, vix, clip=3.0):
    """fit_ou_with_exog 와 동일한 전처리로 설계행렬을 만든다."""
    g = np.asarray(gap, dtype=float).flatten()
    si_z = zscore_clip(np.asarray(si, dtype=float).flatten(), clip=clip)
    vix_z = zscore_clip(np.asarray(vix, dtype=float).flatten(), clip=clip)
    return {
        "delta_g": np.diff(g),
        "g_lag": g[:-1],
        "si_lag": si_z[:-1],
        "vix_lag": vix_z[:-1],
    }


def loglik(params, d):
    """벌점 없는 순수 로그우도. params = [kappa, mu, sigma0, delta1, delta2, nu]"""
    k, m, s0, d1, d2, nu = params
    if s0 <= 0 or k <= 0 or not (NU_MIN <= nu <= NU_MAX):
        return -np.inf
    sigma_t = s0 * np.exp(d1 * d["si_lag"] + d2 * d["vix_lag"])
    if not np.all(np.isfinite(sigma_t)) or np.any(sigma_t <= 0):
        return -np.inf
    mean_t = k * (m - d["g_lag"])
    ll = student_t.logpdf(d["delta_g"], df=nu, loc=mean_t, scale=sigma_t / _t_scale(nu))
    return float(np.sum(ll)) if np.all(np.isfinite(ll)) else -np.inf


def fit_mle(d, x0, fix=None):
    """벌점 없는 MLE. fix={인덱스: 값} 으로 일부 모수를 제약할 수 있다."""
    fix = fix or {}
    free = [i for i in range(6) if i not in fix]

    def unpack(theta):
        p = np.empty(6)
        for i, v in fix.items():
            p[i] = v
        for j, i in enumerate(free):
            p[i] = theta[j]
        return p

    def neg(theta):
        v = loglik(unpack(theta), d)
        return 1e10 if not np.isfinite(v) else -v

    bounds_all = [(1e-6, 10.0), (None, None), (1e-8, None),
                  (-2.0, 2.0), (-2.0, 2.0), (NU_MIN, NU_MAX)]
    res = minimize(neg, x0=[x0[i] for i in free], method="L-BFGS-B",
                   bounds=[bounds_all[i] for i in free])
    p = unpack(res.x)
    return p, loglik(p, d), res.success


def hessian_se(p, d, free_idx):
    """중앙차분 헤시안 -> 관측 정보행렬 -> 표준오차."""
    n = len(free_idx)
    # 모수별 스케일에 비례한 스텝 (절대 스케일 차이가 커서 고정 스텝은 부적절)
    h = np.array([max(abs(p[i]) * 1e-4, 1e-7) for i in free_idx])

    def f(theta):
        q = p.copy()
        for j, i in enumerate(free_idx):
            q[i] = theta[j]
        return loglik(q, d)

    base = np.array([p[i] for i in free_idx])
    H = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            ea = np.zeros(n); ea[a] = h[a]
            eb = np.zeros(n); eb[b] = h[b]
            val = (f(base + ea + eb) - f(base + ea - eb)
                   - f(base - ea + eb) + f(base - ea - eb)) / (4 * h[a] * h[b])
            H[a, b] = H[b, a] = val
    try:
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(n, np.nan)
        cov = np.full((n, n), np.nan)
    return se, cov, H


# ──────────────────────────────────────────────────────────────
# 부트스트랩
# ──────────────────────────────────────────────────────────────
def parametric_bootstrap(p, d, B=500, seed=42):
    """적합된 모형에서 경로를 재생성하고 다시 적합해 추정치 분포를 얻는다.

    외생변수(si_lag, vix_lag)는 실제 값을 그대로 쓰고 혁신항만 다시 뽑는다.
    즉 "이 외생변수 아래에서 이 모형이 맞다면 delta2 추정치가 얼마나 흔들리는가"를 본다.
    """
    k, m, s0, d1, d2, nu = p
    rng = np.random.default_rng(seed)
    sigma_t = s0 * np.exp(d1 * d["si_lag"] + d2 * d["vix_lag"])
    T = len(d["delta_g"])
    g0 = d["g_lag"][0]

    out = []
    for _ in range(B):
        eps = standardized_t(nu, size=T, rng=rng)
        g = np.empty(T + 1)
        g[0] = g0
        for t in range(T):
            g[t + 1] = g[t] + k * (m - g[t]) + sigma_t[t] * eps[t]
        db = {"delta_g": np.diff(g), "g_lag": g[:-1],
              "si_lag": d["si_lag"], "vix_lag": d["vix_lag"]}
        try:
            pb, _, ok = fit_mle(db, x0=p)
            if ok and np.all(np.isfinite(pb)):
                out.append(pb)
        except Exception:
            continue
    return np.array(out)


# ──────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore")
    # 본 파이프라인과 같은 NAV 그리드를 쓴다 (결정 11).
    df = align_to_nav_grid(load_nav_exog_and_returns(), load_gap_exog())
    gap, si, vix = df["etf_premium"], df["value"], df["btc_volatility"]
    d = build_design(gap, si, vix)
    T = len(d["delta_g"])

    out_dir = _root / "results" / "discussion"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 78)
    P("GAP 모형 delta2 진단")
    P("=" * 78)
    P(f"표본: {df['Date'].iloc[0]} ~ {df['Date'].iloc[-1]}, 차분 관측 T = {T}")

    # ── 운영 추정치 (벌점 포함) ────────────────────────────────
    op = fit_ou_with_exog(gap, si, vix, dist="t")
    op_vec = np.array([op["kappa"], op["mu"], op["sigma0"],
                       op["delta1"], op["delta2"], op["nu"]])
    P("\n[운영 추정치] 파이프라인이 쓰는 값 (능형 벌점 0.01*(d1^2+d2^2) 포함)")
    for n_, v in zip(PARAM_NAMES, op_vec):
        P(f"    {n_:8} = {v:+.6f}")
    P(f"    로그우도(벌점 제외) = {loglik(op_vec, d):.4f}")

    # ── 1. MLE + 표준오차 ─────────────────────────────────────
    P("\n" + "-" * 78)
    P("1. 벌점 없는 MLE 와 표준오차")
    P("-" * 78)
    mle, ll_full, ok = fit_mle(d, x0=op_vec)
    P(f"  수렴: {ok}    로그우도 = {ll_full:.4f}")
    se, cov, _ = hessian_se(mle, d, free_idx=list(range(6)))

    P(f"\n  {'모수':8}{'추정치':>13}{'표준오차':>13}{'t값':>10}{'p값':>10}   {'95% 신뢰구간':>26}")
    rows = []
    for i, n_ in enumerate(PARAM_NAMES):
        est, s_ = mle[i], se[i]
        tval = est / s_ if s_ and np.isfinite(s_) and s_ > 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(tval))) if np.isfinite(tval) else np.nan
        lo, hi = (est - 1.96 * s_, est + 1.96 * s_) if np.isfinite(s_) else (np.nan, np.nan)
        P(f"  {n_:8}{est:+13.6f}{s_:13.6f}{tval:10.2f}{pval:10.4f}   [{lo:+.6f}, {hi:+.6f}]")
        rows.append({"param": n_, "mle": est, "se_hessian": s_, "t": tval,
                     "p": pval, "ci_lo": lo, "ci_hi": hi,
                     "operational": op_vec[i]})

    # ── 2. delta2 유의성 ──────────────────────────────────────
    P("\n" + "-" * 78)
    P("2. delta2 의 통계적 유의성")
    P("-" * 78)
    i2 = PARAM_NAMES.index("delta2")
    t2, p2 = rows[i2]["t"], rows[i2]["p"]
    P(f"  MLE delta2 = {mle[i2]:+.6f}   t = {t2:.2f}   p = {p2:.4f}")
    P(f"  95% 신뢰구간 [{rows[i2]['ci_lo']:+.6f}, {rows[i2]['ci_hi']:+.6f}]"
      f"  -> 0 포함 {'예' if rows[i2]['ci_lo'] < 0 < rows[i2]['ci_hi'] else '아니오'}")
    P(f"  판정: {'유의 (5%)' if np.isfinite(p2) and p2 < 0.05 else '유의하지 않음 (5%)'}")

    # ── 3. SI 와 RV 의 상관 ───────────────────────────────────
    P("\n" + "-" * 78)
    P("3. SI 와 RV 의 상관 (공선성 확인)")
    P("-" * 78)
    r_raw = stats.pearsonr(si.values, vix.values)
    s_raw = stats.spearmanr(si.values, vix.values)
    r_used = stats.pearsonr(d["si_lag"], d["vix_lag"])
    P(f"  원자료      Pearson r = {r_raw[0]:+.4f} (p={r_raw[1]:.2e})   "
      f"Spearman = {s_raw[0]:+.4f}")
    P(f"  모형 투입값 Pearson r = {r_used[0]:+.4f} (p={r_used[1]:.2e})   (z-표준화·클리핑 후)")
    vif = 1.0 / (1.0 - r_used[0] ** 2)
    P(f"  VIF = {vif:.2f}  ({'공선성 문제 없음' if vif < 5 else '공선성 우려'})")
    if np.isfinite(cov[3, 4]) and se[3] > 0 and se[4] > 0:
        P(f"  추정치 간 상관 corr(delta1_hat, delta2_hat) = {cov[3,4]/(se[3]*se[4]):+.4f}")

    # ── 4. LR 검정 ────────────────────────────────────────────
    P("\n" + "-" * 78)
    P("4. delta2 = 0 제약 대 자유 추정 (우도비 검정)")
    P("-" * 78)
    _, ll_r, ok_r = fit_mle(d, x0=op_vec, fix={4: 0.0})
    lr = 2 * (ll_full - ll_r)
    p_lr = 1 - stats.chi2.cdf(lr, df=1)
    P(f"  자유 추정  로그우도 = {ll_full:.4f}")
    P(f"  delta2=0   로그우도 = {ll_r:.4f}   (수렴 {ok_r})")
    P(f"  LR = 2*(차이) = {lr:.4f}   chi2(1) p = {p_lr:.4f}")
    P(f"  판정: {'delta2 를 자유롭게 두는 편이 유의하게 낫다' if p_lr < 0.05 else 'delta2 를 0으로 제약해도 우도가 유의하게 나빠지지 않는다'}")

    # ── 5. 부분표본 안정성 ────────────────────────────────────
    P("\n" + "-" * 78)
    P("5. 부분표본에서 delta2 부호가 유지되는가")
    P("-" * 78)
    half = T // 2
    subs = {"전반부": slice(0, half), "후반부": slice(half, T)}
    P(f"  {'구간':8}{'n':>6}{'delta1':>12}{'delta2':>12}{'nu':>9}{'sigma0':>11}")
    sub_rows = []
    for lab, sl in subs.items():
        ds = {k_: v[sl] for k_, v in d.items()}
        ps, _, oks = fit_mle(ds, x0=mle)
        P(f"  {lab:8}{len(ds['delta_g']):6d}{ps[3]:+12.5f}{ps[4]:+12.5f}{ps[5]:9.2f}{ps[2]:11.6f}")
        sub_rows.append({"subsample": lab, "n": len(ds["delta_g"]),
                         "delta1": ps[3], "delta2": ps[4], "nu": ps[5], "sigma0": ps[2]})
    signs = [np.sign(r["delta2"]) for r in sub_rows]
    P(f"  전체표본 delta2 부호 = {'음수' if mle[4] < 0 else '양수'},  "
      f"부분표본 부호 일치: {'예' if len(set(signs)) == 1 and signs[0] == np.sign(mle[4]) else '아니오'}")

    # ── 부트스트랩 ────────────────────────────────────────────
    P("\n" + "-" * 78)
    P("보강: 모수 부트스트랩 (B=500) — 헤시안 표준오차 교차확인")
    P("-" * 78)
    bs = parametric_bootstrap(mle, d, B=500, seed=42)
    P(f"  성공한 재표본: {len(bs)}/500")
    if len(bs) > 50:
        P(f"\n  {'모수':8}{'부트 평균':>13}{'부트 SE':>12}{'헤시안 SE':>12}{'2.5%':>12}{'97.5%':>12}")
        for i, n_ in enumerate(PARAM_NAMES):
            lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
            P(f"  {n_:8}{bs[:,i].mean():+13.6f}{bs[:,i].std():12.6f}"
              f"{se[i]:12.6f}{lo:+12.6f}{hi:+12.6f}")
            rows[i]["se_bootstrap"] = float(bs[:, i].std())
            rows[i]["boot_lo"] = float(lo)
            rows[i]["boot_hi"] = float(hi)
        frac = float(np.mean(bs[:, 4] < 0))
        P(f"\n  부트스트랩에서 delta2 < 0 인 비율 = {frac*100:.1f}%")
        P(f"  (모형이 참일 때 이 방향이 얼마나 안정적으로 재현되는지)")

    pd.DataFrame(rows).to_csv(out_dir / "gap_delta2_params.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(sub_rows).to_csv(out_dir / "gap_delta2_subsamples.csv",
                                  index=False, encoding="utf-8-sig")
    (out_dir / "gap_delta2_report.txt").write_text("\n".join(lines), encoding="utf-8")
    P(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
