"""
Bai-Perron (1998, 2003) 다중 구조 변화 검정 모듈
==================================================
ASVI 변환된 시계열에 대해 구간 분할(structural break) 을 수행한다.

모형: y_t = μ_j + ε_t  (순수 평균 변화 모형)

검정 절차:
  1. supF(m)    : 정확히 m개 break 존재 여부 (vs 0개)
  2. UDmax      : break 개수 미지 조건 — supF(m) 의 최대값
  3. WDmax      : UDmax 의 가중치 버전
  4. F(l+1|l)   : 순차 검정 — l개 break 파티션 내 각 구간에서 추가 break 탐색
  5. BIC        : 최적 break 개수 결정

임계값 출처:
  Bai & Perron (2003), "Critical Values for Multiple Structural Change Tests",
  Econometrics Journal 6(1), 72-78.
  파라미터: q=1 (평균만 이동), p=0, ε=0.10 (trimming)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════
# 임계값 테이블 (Bai & Perron 2003)
# q=1, p=0, ε=0.10
# ══════════════════════════════════════════════

# supF(m) 임계값 — H0: 0개 break vs H1: 정확히 m개 break
# (m, 유의수준) → 임계값
SUPF_CV: Dict[Tuple[int, float], float] = {
    (1, 0.10): 7.17,  (1, 0.05):  8.58, (1, 0.025): 10.04, (1, 0.01): 11.97,
    (2, 0.10): 5.49,  (2, 0.05):  7.22, (2, 0.025):  8.51, (2, 0.01): 10.22,
    (3, 0.10): 4.09,  (3, 0.05):  5.96, (3, 0.025):  7.17, (3, 0.01):  8.98,
    (4, 0.10): 3.59,  (4, 0.05):  4.99, (4, 0.025):  6.05, (4, 0.01):  7.56,
    (5, 0.10): 3.10,  (5, 0.05):  4.28, (5, 0.025):  5.26, (5, 0.01):  6.68,
}

# UDmax 임계값 (유의수준 → 임계값)
UDMAX_CV: Dict[float, float] = {
    0.10: 7.56, 0.05: 8.88, 0.025: 10.27, 0.01: 11.97,
}

# WDmax 임계값
WDMAX_CV: Dict[float, float] = {
    0.10: 8.36, 0.05: 9.91, 0.025: 11.39, 0.01: 13.45,
}

# 순차 검정 F(l+1|l) 임계값 — H0: l개 break vs H1: l+1개 break
# (l_breaks_under_null, 유의수준) → 임계값
SEQ_CV: Dict[Tuple[int, float], float] = {
    (0, 0.10):  7.17, (0, 0.05):  8.58, (0, 0.025): 10.04, (0, 0.01): 11.97,
    (1, 0.10):  7.97, (1, 0.05): 10.13, (1, 0.025): 12.06, (1, 0.01): 14.67,
    (2, 0.10):  8.95, (2, 0.05): 11.14, (2, 0.025): 13.24, (2, 0.01): 16.19,
    (3, 0.10):  9.73, (3, 0.05): 11.83, (3, 0.025): 14.10, (3, 0.01): 16.97,
    (4, 0.10):  9.73, (4, 0.05): 12.25, (4, 0.025): 14.46, (4, 0.01): 17.20,
}

# WDmax 가중치 (BP 2003 Table 2, m별 균형 조정)
WDMAX_W: Dict[int, float] = {1: 1.00, 2: 1.30, 3: 1.52, 4: 1.78, 5: 2.11}


# ══════════════════════════════════════════════
# 내부 헬퍼
# ══════════════════════════════════════════════

def _newey_west_var(resid: np.ndarray, bw: Optional[int] = None) -> float:
    """Bartlett 커널 Newey-West HAC 장기분산 (per-observation)."""
    T = len(resid)
    if bw is None:
        bw = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    omega = np.mean(resid ** 2)
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1)          # Bartlett 가중치
        gamma_l = np.mean(resid[lag:] * resid[:-lag])
        omega += 2.0 * w * gamma_l
    return max(omega, 1e-14)


def _seg_rss_mean(
    values: np.ndarray,
    cum: np.ndarray,
    cum_sq: np.ndarray,
    i: int,
    j: int,
) -> Tuple[float, float]:
    """구간 [i, j](0-indexed, inclusive)의 RSS 와 평균을 반환."""
    n    = j - i + 1
    s    = cum[j + 1] - cum[i]
    sq   = cum_sq[j + 1] - cum_sq[i]
    mean = s / n
    rss  = max(0.0, sq - s * s / n)
    return rss, mean


def _optimal_partition_dp(
    values: np.ndarray,
    cum: np.ndarray,
    cum_sq: np.ndarray,
    m_breaks: int,
    T: int,
    h: int,
) -> Tuple[List[int], float]:
    """Bai-Perron 동적 계획법으로 최적 m-break 파티션 탐색.

    Returns
    -------
    breaks : list of int
        각 구간의 끝 인덱스 (0-indexed). 길이 = m_breaks.
        (구간 j 는 [breaks[j-1]+1, breaks[j]], 마지막은 [breaks[-1]+1, T-1])
    total_rss : float
        최적 파티션의 총 RSS.
    """
    if m_breaks == 0:
        rss0, _ = _seg_rss_mean(values, cum, cum_sq, 0, T - 1)
        return [], rss0

    INF = np.inf
    # dp[k, j]: k개 break 를 사용하여 [0..j] 를 분할했을 때 최소 RSS
    # bp[k, j]: 위 최적에서 k-번째 구간의 끝 인덱스 (= i-1)
    dp = np.full((m_breaks + 1, T), INF)
    bp = np.full((m_breaks + 1, T), -1, dtype=np.int32)

    # k=0 (1 구간): [0..j]
    for j in range(h - 1, T):
        rss, _ = _seg_rss_mean(values, cum, cum_sq, 0, j)
        dp[0, j] = rss

    # k=1..m_breaks
    for k in range(1, m_breaks + 1):
        j_min = (k + 1) * h - 1
        for j in range(j_min, T):
            i_min = k * h
            i_max = j - h + 1
            if i_min > i_max:
                continue

            # 벡터화: 마지막 구간 [i..j] (i ∈ [i_min, i_max])
            I     = np.arange(i_min, i_max + 1)
            n_seg = j - I + 1
            s_ij  = cum[j + 1] - cum[I]
            sq_ij = cum_sq[j + 1] - cum_sq[I]
            rss_ij = np.maximum(0.0, sq_ij - s_ij ** 2 / n_seg)

            total = dp[k - 1, I - 1] + rss_ij
            best  = int(np.argmin(total))

            dp[k, j] = total[best]
            bp[k, j] = I[best] - 1   # 이전 구간의 끝 = i-1

    if dp[m_breaks, T - 1] == INF:
        return [], INF

    # 역추적 (backtrack)
    breaks: List[int] = []
    pos = T - 1
    for k in range(m_breaks, 0, -1):
        b = int(bp[k, pos])
        breaks.append(b)
        pos = b
    breaks.reverse()

    return breaks, float(dp[m_breaks, T - 1])


def _max_one_break_F(
    values: np.ndarray,
    cum: np.ndarray,
    cum_sq: np.ndarray,
    a: int,
    b: int,
    h: int,
    omega_sq: float,
) -> Tuple[float, int]:
    """구간 [a, b] 내 단일 break 에 대한 최대 F 통계량과 최적 break 위치."""
    rss_full, _ = _seg_rss_mean(values, cum, cum_sq, a, b)
    max_F  = 0.0
    best_k = a + h

    if b - a + 1 < 2 * h:
        return 0.0, best_k

    k_arr    = np.arange(a + h, b - h + 2)
    # 왼쪽 구간 [a, k-1]
    n_l   = k_arr - a
    s_l   = cum[k_arr] - cum[a]
    sq_l  = cum_sq[k_arr] - cum_sq[a]
    rss_l = np.maximum(0.0, sq_l - s_l ** 2 / n_l)
    # 오른쪽 구간 [k, b]
    n_r   = b - k_arr + 1
    s_r   = cum[b + 1] - cum[k_arr]
    sq_r  = cum_sq[b + 1] - cum_sq[k_arr]
    rss_r = np.maximum(0.0, sq_r - s_r ** 2 / n_r)

    F_arr  = (rss_full - rss_l - rss_r) / omega_sq
    best   = int(np.argmax(F_arr))
    max_F  = float(F_arr[best])
    best_k = int(k_arr[best])

    return max_F, best_k


# ══════════════════════════════════════════════
# 결과 데이터 클래스
# ══════════════════════════════════════════════

@dataclass
class SegmentInfo:
    """단일 구간의 추정 결과."""
    segment_idx: int         # 구간 번호 (1부터)
    start_idx:   int         # 시작 인덱스 (0-based)
    end_idx:     int         # 종료 인덱스 (0-based)
    start_date:  object      # 시작 날짜
    end_date:    object      # 종료 날짜
    n_obs:       int         # 관측치 수
    mean:        float       # 구간 평균 μ_j
    ci_lower:    float       # 95% 신뢰구간 하한
    ci_upper:    float       # 95% 신뢰구간 상한
    rss:         float       # 구간 내 RSS


@dataclass
class BaiPerronResult:
    """Bai-Perron 검정 결과 전체 컨테이너."""

    variable_name: str
    series:        pd.Series

    # BIC 최적 break 개수 및 날짜
    optimal_m:    int
    break_dates:  List[object]
    segments:     List[SegmentInfo]

    # 검정 통계량
    supF:     Dict[int, float]    # supF(m) 통계량
    supF_cv:  Dict[int, float]    # supF(m) 5% 임계값
    udmax:    float
    udmax_cv: float
    wdmax:    float
    wdmax_cv: float
    seqF:     Dict[int, float]    # F(l+1|l) 통계량
    seqF_cv:  Dict[int, float]    # F(l+1|l) 5% 임계값

    # BIC 테이블
    bic_table: pd.DataFrame

    # 파라미터
    m_max:         int
    trim:          float
    sig_level:     float
    hac_bw:        int
    omega_sq:      float

    # 내부: 모든 m 별 최적 break indices (구간 끝 인덱스, 0-based)
    all_breaks:    Dict[int, List[int]] = field(repr=False)
    all_ssr:       Dict[int, float]     = field(repr=False)

    # ──────────────────────────────────────────
    def print_summary(self) -> None:
        print("=" * 68)
        print(f"  Bai-Perron Structural Break Test: [{self.variable_name}]")
        print(f"  T={len(self.series)}, m_max={self.m_max}, "
              f"trim={self.trim}, sig={self.sig_level}")
        print(f"  HAC: omega^2={self.omega_sq:.6f}  (BW={self.hac_bw})")
        print("=" * 68)

        print(f"\n{'-- supF(m) Global Test ':-<48}  (5% CV)")
        for m in range(1, self.m_max + 1):
            s  = self.supF.get(m, np.nan)
            cv = self.supF_cv.get(m, np.nan)
            mk = "  * Sig." if s > cv else ""
            print(f"  supF({m}) = {s:8.3f}  |  CV = {cv:6.3f}{mk}")

        print(f"\n  UDmax = {self.udmax:8.3f}  |  CV = {self.udmax_cv:.3f}"
              f"{'  * Sig.' if self.udmax > self.udmax_cv else ''}")
        print(f"  WDmax = {self.wdmax:8.3f}  |  CV = {self.wdmax_cv:.3f}"
              f"{'  * Sig.' if self.wdmax > self.wdmax_cv else ''}")

        print(f"\n{'-- Sequential Test F(l+1|l) ':-<48}  (5% CV)")
        for l in range(len(self.seqF)):
            s  = self.seqF.get(l, np.nan)
            cv = self.seqF_cv.get(l, np.nan)
            mk = "  * Sig." if s > cv else ""
            print(f"  F({l+1}|{l}) = {s:8.3f}  |  CV = {cv:6.3f}{mk}")

        print(f"\n{'-- BIC Table ':-<48}")
        print(self.bic_table.to_string(float_format="{:.4f}".format))

        print(f"\n  >> BIC Optimal breaks: m = {self.optimal_m}")

        if self.break_dates:
            print(f"\n{'-- Optimal Break Dates ':-<48}")
            for i, dt in enumerate(self.break_dates, 1):
                print(f"  Break {i}: {dt}")

        if self.segments:
            print(f"\n{'-- Segment Means mu_j and 95% CI ':-<48}")
            hdr = f"  {'Seg':^4}  {'Period':^38}  {'n':>5}  {'mu_j':>9}  {'95% CI':<22}"
            print(hdr)
            print(f"  {'-'*82}")
            for s in self.segments:
                ci_str = f"[{s.ci_lower:7.4f}, {s.ci_upper:7.4f}]"
                period = f"{s.start_date} ~ {s.end_date}"
                print(f"  {s.segment_idx:^4}  {period:<38}  {s.n_obs:>5}  "
                      f"{s.mean:>9.4f}  {ci_str}")
        print("=" * 68)


# ══════════════════════════════════════════════
# 핵심 검정 함수
# ══════════════════════════════════════════════

def fit_bai_perron(
    series: pd.Series,
    m_max: int = 5,
    trim: float = 0.10,
    sig_level: float = 0.05,
    hac_bw: Optional[int] = None,
) -> BaiPerronResult:
    """Bai-Perron (1998, 2003) 다중 구조 변화 검정을 수행한다.

    Parameters
    ----------
    series : pd.Series
        ASVI 변환된 주별 시계열. index 는 DatetimeIndex 권장.
    m_max : int
        최대 break 개수 (기본 5).
    trim : float
        최소 구간 비율 ε (기본 0.10). 최소 구간 길이 h = floor(ε·T).
    sig_level : float
        유의수준 (기본 0.05). 임계값 룩업에 사용.
    hac_bw : int, optional
        Newey-West 대역폭. None 이면 Andrews(1991) 자동 선택.

    Returns
    -------
    BaiPerronResult
    """
    ser    = series.dropna()
    values = ser.values.astype(float)
    T      = len(values)
    index  = ser.index
    h      = max(2, int(np.floor(trim * T)))

    if T < (m_max + 1) * h:
        raise ValueError(
            f"표본이 부족합니다: T={T}, m_max={m_max}, h={h}. "
            f"최소 T ≥ {(m_max + 1) * h} 필요."
        )

    # ── 누적합 precompute ─────────────────────────────────────
    cum    = np.concatenate([[0.0], np.cumsum(values)])
    cum_sq = np.concatenate([[0.0], np.cumsum(values ** 2)])

    # ── HAC 장기분산 (귀무모형 잔차 기준) ─────────────────────
    null_resid = values - values.mean()
    if hac_bw is None:
        bw = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    else:
        bw = hac_bw
    omega_sq = _newey_west_var(null_resid, bw=bw)

    # ── 1. 모든 m 에 대해 최적 partition 탐색 (DP) ───────────
    all_breaks: Dict[int, List[int]] = {}
    all_ssr:    Dict[int, float]     = {}

    rss0, _ = _seg_rss_mean(values, cum, cum_sq, 0, T - 1)
    all_ssr[0] = rss0

    for m in range(1, m_max + 1):
        bp, ssr_m = _optimal_partition_dp(values, cum, cum_sq, m, T, h)
        all_breaks[m] = bp
        all_ssr[m]    = ssr_m

    # ── 2. supF(m) 통계량 ─────────────────────────────────────
    # F(m) = (SSR_0 - SSR_opt_m) / (m × ω²)
    # ω² = per-observation HAC 분산, SSR ~ O(T·ω²) ⟹ F = O(1) under H0
    supF: Dict[int, float] = {}
    for m in range(1, m_max + 1):
        diff    = all_ssr[0] - all_ssr[m]
        supF[m] = diff / (m * omega_sq)

    supF_cv = {m: SUPF_CV.get((m, sig_level), np.nan) for m in range(1, m_max + 1)}

    # ── 3. UDmax / WDmax ──────────────────────────────────────
    udmax    = float(max(supF.values()))
    wdmax    = float(max(WDMAX_W.get(m, 1.0) * supF[m] for m in range(1, m_max + 1)))
    udmax_cv = UDMAX_CV.get(sig_level, np.nan)
    wdmax_cv = WDMAX_CV.get(sig_level, np.nan)

    # ── 4. 순차 검정 F(l+1|l) ────────────────────────────────
    # l-break 최적 파티션의 각 구간에서 단일 break 탐색 후 최대값
    seqF:    Dict[int, float] = {}
    seqF_cv: Dict[int, float] = {}

    for l in range(m_max):
        if l == 0:
            F_stat, _ = _max_one_break_F(values, cum, cum_sq, 0, T - 1, h, omega_sq)
            seqF[l] = F_stat
        else:
            boundaries = [0] + [b + 1 for b in all_breaks[l]] + [T]
            seg_Fs = []
            for j in range(len(boundaries) - 1):
                a, b = boundaries[j], boundaries[j + 1] - 1
                if b - a + 1 >= 2 * h:
                    F_seg, _ = _max_one_break_F(
                        values, cum, cum_sq, a, b, h, omega_sq
                    )
                    seg_Fs.append(F_seg)
            seqF[l] = float(max(seg_Fs)) if seg_Fs else 0.0

        seqF_cv[l] = SEQ_CV.get((l, sig_level), np.nan)

    # ── 5. BIC ────────────────────────────────────────────────
    # BIC(m) = T·log(SSR_m/T) + (m+1)·log(T)
    # 패널티: (m+1) 개 평균 파라미터 × log(T)
    bic_rows = []
    for m in range(m_max + 1):
        ssr_m = all_ssr[m]
        bic_m = T * np.log(ssr_m / T) + (m + 1) * np.log(T)
        bic_rows.append({"m": m, "SSR": ssr_m, "log(SSR/T)": np.log(ssr_m / T), "BIC": bic_m})
    bic_df     = pd.DataFrame(bic_rows).set_index("m")
    optimal_m  = int(bic_df["BIC"].idxmin())

    # ── 6. 최적 break 날짜 및 구간 정보 ─────────────────────
    break_idx = all_breaks.get(optimal_m, [])
    break_dates = [index[i] for i in break_idx]

    boundaries = [0] + [i + 1 for i in break_idx] + [T]
    z_alpha    = stats.norm.ppf(1 - 0.025)    # 1.96
    segments   = []
    for j in range(len(boundaries) - 1):
        a, b    = boundaries[j], boundaries[j + 1] - 1
        n_seg   = b - a + 1
        rss_s, mu_s = _seg_rss_mean(values, cum, cum_sq, a, b)
        se_s    = np.sqrt(omega_sq / n_seg)
        segments.append(SegmentInfo(
            segment_idx = j + 1,
            start_idx   = a,
            end_idx     = b,
            start_date  = index[a],
            end_date    = index[b],
            n_obs       = n_seg,
            mean        = float(mu_s),
            ci_lower    = float(mu_s - z_alpha * se_s),
            ci_upper    = float(mu_s + z_alpha * se_s),
            rss         = float(rss_s),
        ))

    return BaiPerronResult(
        variable_name = str(series.name) if series.name else "series",
        series        = ser,
        optimal_m     = optimal_m,
        break_dates   = break_dates,
        segments      = segments,
        supF          = supF,
        supF_cv       = supF_cv,
        udmax         = udmax,
        udmax_cv      = udmax_cv,
        wdmax         = wdmax,
        wdmax_cv      = wdmax_cv,
        seqF          = seqF,
        seqF_cv       = seqF_cv,
        bic_table     = bic_df,
        m_max         = m_max,
        trim          = trim,
        sig_level     = sig_level,
        hac_bw        = bw,
        omega_sq      = omega_sq,
        all_breaks    = all_breaks,
        all_ssr       = all_ssr,
    )


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def plot_bai_perron(
    result: BaiPerronResult,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bai-Perron 결과 4-패널 시각화.

    패널:
      [1] 시계열 + break 수직선 + 구간 평균
      [2] 구간 평균 μ_j 및 95% CI (수평 에러바)
      [3] supF(m) 검정 통계량 vs 임계값 (막대 + 점선)
      [4] BIC 곡선 (break 개수별)

    Parameters
    ----------
    result : BaiPerronResult
    figsize : tuple
    save_path : str | None
        지정하면 PNG 저장.
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[2.5, 1.2, 1.5, 1.5],
        hspace=0.55, wspace=0.35,
    )

    ax_ts  = fig.add_subplot(gs[0, :])   # 전체 너비 — 시계열
    ax_ci  = fig.add_subplot(gs[1, :])   # 전체 너비 — 구간 평균 CI
    ax_sup = fig.add_subplot(gs[2, 0])   # supF 막대
    ax_seq = fig.add_subplot(gs[2, 1])   # sequential F 막대
    ax_bic = fig.add_subplot(gs[3, :])   # BIC 곡선

    ser    = result.series
    name   = result.variable_name

    # ── 패널 1: 시계열 + break lines + 구간 평균 ────────────
    # 시계열 레이블: 컬럼명에서 prefix 제거해 범례를 깔끔하게
    short_label = name.replace("[ASVI^2] asvi_", "").replace("asvi_", "")

    ax_ts.plot(ser.index, ser.values, color="#555555", linewidth=1.0,
               alpha=0.7, label=short_label)

    for i, seg in enumerate(result.segments):
        col = _COLORS[i % len(_COLORS)]
        idx = ser.index[seg.start_idx: seg.end_idx + 1]
        ax_ts.hlines(
            seg.mean, idx[0], idx[-1],
            color=col, linewidth=2.2, linestyle="-",
            label=f"mu_{seg.segment_idx}={seg.mean:.3f}",
        )
        ax_ts.fill_between(
            idx,
            [seg.ci_lower] * len(idx),
            [seg.ci_upper] * len(idx),
            color=col, alpha=0.10,
        )

    for dt in result.break_dates:
        ax_ts.axvline(x=dt, color="#d62728", linewidth=1.5,
                      linestyle="--", alpha=0.8)

    # y축 레이블: ASVI² 여부를 명시
    is_squared = "[ASVI^2]" in name
    y_label    = "ASVI^2  (= Variance Proxy)" if is_squared else "Value"

    ax_ts.set_title(
        f"{short_label}  |  Bai-Perron Optimal Breaks: m={result.optimal_m}  (BIC)"
        + ("  [target: ASVI^2, not ASVI]" if is_squared else ""),
        fontsize=10,
    )
    ax_ts.set_ylabel(y_label)
    ax_ts.legend(fontsize=8, loc="upper right", ncol=3)
    ax_ts.grid(True, linestyle="--", alpha=0.3)

    # ── 패널 2: 구간 평균 및 95% CI ─────────────────────────
    y_pos  = range(len(result.segments))
    means  = [s.mean    for s in result.segments]
    lows   = [s.mean - s.ci_lower  for s in result.segments]
    highs  = [s.ci_upper - s.mean  for s in result.segments]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(result.segments))]

    ax_ci.barh(
        y_pos, means,
        xerr=[lows, highs],
        color=colors, alpha=0.7,
        error_kw=dict(ecolor="black", capsize=4, linewidth=1.2),
    )
    ax_ci.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    ax_ci.set_yticks(list(y_pos))
    ax_ci.set_yticklabels([f"Seg {s.segment_idx}\n({s.start_date}~{s.end_date})"
                           for s in result.segments], fontsize=7)
    ax_ci.set_xlabel("mu_j (95% CI)")
    ax_ci.set_title("Segment Mean mu_j with 95% CI", fontsize=10)
    ax_ci.grid(True, axis="x", linestyle="--", alpha=0.35)

    # ── 패널 3: supF(m) 검정 ──────────────────────────────
    m_vals   = list(result.supF.keys())
    f_vals   = [result.supF[m]    for m in m_vals]
    cv_vals  = [result.supF_cv[m] for m in m_vals]

    bar_colors = [
        "#d62728" if f > cv else "#aec7e8"
        for f, cv in zip(f_vals, cv_vals)
    ]
    ax_sup.bar(m_vals, f_vals, color=bar_colors, alpha=0.85, width=0.5)
    ax_sup.step(
        [m - 0.3 for m in m_vals] + [m_vals[-1] + 0.3],
        cv_vals + [cv_vals[-1]],
        color="#ff7f0e", linewidth=1.5, linestyle="--",
        where="post", label="5% CV",
    )
    ax_sup.set_xticks(m_vals)
    ax_sup.set_xlabel("m (# breaks)")
    ax_sup.set_ylabel("Test Statistic")
    ax_sup.set_title("supF(m) Test", fontsize=10)
    ax_sup.legend(fontsize=8)
    ax_sup.grid(True, linestyle="--", alpha=0.35)

    # ── 패널 4: 순차 F(l+1|l) ──────────────────────────────
    l_vals   = list(result.seqF.keys())
    fs_vals  = [result.seqF[l]    for l in l_vals]
    cvs_vals = [result.seqF_cv[l] for l in l_vals]
    xlabels  = [f"F({l+1}|{l})" for l in l_vals]

    bar_colors2 = [
        "#d62728" if f > cv else "#aec7e8"
        for f, cv in zip(fs_vals, cvs_vals)
    ]
    ax_seq.bar(range(len(l_vals)), fs_vals, color=bar_colors2, alpha=0.85, width=0.5)
    ax_seq.step(
        [x - 0.3 for x in range(len(l_vals))] + [len(l_vals) - 0.7],
        cvs_vals + [cvs_vals[-1]],
        color="#ff7f0e", linewidth=1.5, linestyle="--",
        where="post", label="5% CV",
    )
    ax_seq.set_xticks(range(len(l_vals)))
    ax_seq.set_xticklabels(xlabels, fontsize=8)
    ax_seq.set_ylabel("Test Statistic")
    ax_seq.set_title("Sequential Test F(l+1|l)", fontsize=10)
    ax_seq.legend(fontsize=8)
    ax_seq.grid(True, linestyle="--", alpha=0.35)

    # ── 패널 5: BIC 곡선 ────────────────────────────────────
    bic = result.bic_table["BIC"]
    ax_bic.plot(bic.index, bic.values, marker="o", color="#2ca02c",
                linewidth=1.5, markersize=6)
    ax_bic.axvline(
        x=result.optimal_m, color="#d62728",
        linewidth=1.5, linestyle="--",
        label=f"Optimal m={result.optimal_m}",
    )
    ax_bic.set_xticks(list(bic.index))
    ax_bic.set_xlabel("# Breaks (m)")
    ax_bic.set_ylabel("BIC")
    ax_bic.set_title("BIC by Number of Breaks", fontsize=10)
    ax_bic.legend(fontsize=9)
    ax_bic.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        f"Bai-Perron Multiple Structural Break Test: {name}",
        fontsize=13, fontweight="bold",
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[BP] Plot saved -> {save_path}")

    return fig


# ══════════════════════════════════════════════
# 다변수 파이프라인
# ══════════════════════════════════════════════

def run_bai_perron_pipeline(
    df: pd.DataFrame,
    asvi_columns: List[str],
    m_max: Union[int, Dict[str, int]] = 5,
    trim: float = 0.10,
    sig_level: float = 0.05,
    hac_bw: Optional[int] = None,
    plot: bool = True,
    save_dir: Optional[str] = None,
) -> Dict[str, BaiPerronResult]:
    """여러 ASVI 컬럼에 대해 Bai-Perron 검정을 일괄 수행한다.

    Parameters
    ----------
    df : pd.DataFrame
        ASVI 변환이 완료된 DataFrame.
    asvi_columns : list of str
        검정 대상 컬럼명 목록.
        예: ["asvi_global_btc_svi", "asvi_domestic_btc_svi", "asvi_btc_volume_krw"]
    m_max : int or dict[str, int]
        최대 break 개수. int 이면 전체 컬럼 공통 적용,
        dict 이면 컬럼별 개별 지정 (예: {"global_btc_svi": 3, "btc_volume_btc": 5}).
    trim : float
        최소 구간 비율 ε.
    sig_level : float
        유의수준.
    hac_bw : int, optional
        HAC 대역폭 (None = 자동).
    plot : bool
        True 이면 각 변수별 시각화.
    save_dir : str, optional
        시각화 저장 디렉토리 (None 이면 저장 안 함).

    Returns
    -------
    dict : {컬럼명 → BaiPerronResult}
    """
    results: Dict[str, BaiPerronResult] = {}

    for col in asvi_columns:
        if col not in df.columns:
            print(f"[BP] Column not found, skipping: {col}")
            continue

        print(f"\n{'='*68}")
        print(f"  Processing: {col}")
        print(f"{'='*68}")

        col_m_max = m_max.get(col, 5) if isinstance(m_max, dict) else m_max
        res = fit_bai_perron(
            df[col],
            m_max     = col_m_max,
            trim      = trim,
            sig_level = sig_level,
            hac_bw    = hac_bw,
        )
        res.print_summary()
        results[col] = res

        if plot:
            save_path = None
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"bp_{col}.png")
            plot_bai_perron(res, save_path=save_path)
            plt.show()

    return results


# ══════════════════════════════════════════════
# 독립 실행 예시
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from preprocessing.asvi_transformer import run_asvi_pipeline

    # ── 실제 데이터 로드 (dataset/train_ver1) ────────────────────────
    _DATA = _ROOT / "dataset" / "train_ver1"

    kp_df  = pd.read_csv(_DATA / "kp_train.csv",  parse_dates=["Date"]).set_index("Date").sort_index()
    gap_df = pd.read_csv(_DATA / "gap_train.csv", parse_dates=["Date"]).set_index("Date").sort_index()

    # 주별 리샘플링
    df_weekly = pd.DataFrame(
        {
            "global_btc_svi":   gap_df["value"].resample("W-MON").last(),
            "domestic_btc_svi": kp_df["bitcoin_kr"].resample("W-MON").last(),
            "btc_volume_btc":   kp_df["volume_btc"].resample("W-MON").sum(),
        }
    ).dropna()

    REAL_MAP = {
        "global_svi":   "global_btc_svi",
        "domestic_svi": "domestic_btc_svi",
        "volume_btc":   "btc_volume_btc",
    }

    # ASVI 변환 (시각화 없이)
    df_asvi, _ = run_asvi_pipeline(
        df=df_weekly, column_map=REAL_MAP, window=4, plot=False
    )
    # ─────────────────────────────────────────────────────────────────

    asvi_cols = [f"asvi_{col}" for col in REAL_MAP.values()]

    all_results = run_bai_perron_pipeline(
        df           = df_asvi,
        asvi_columns = asvi_cols,
        m_max        = 5,
        trim         = 0.10,
        sig_level    = 0.05,
        hac_bw       = None,
        plot         = True,
        save_dir     = None,
    )
