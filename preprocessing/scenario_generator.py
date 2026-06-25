"""
시나리오 생성기 (Scenario Generator)
=====================================
Gaussian HMM 레짐 기반 시나리오 시계열 생성 파이프라인.

[설치]
  pip install hmmlearn

[사용 흐름]
  Option A — 기존 파이프라인 결과 재활용 (권장):
    pipeline_out = run_pipeline.main()
    hmm_results  = from_pipeline_results(pipeline_out)

  Option B — 독립 학습:
    hmm_results = {var: fit_scenario_hmm(series, n_states) for ...}

  공통 이후:
    scenario = define_scenario({"Global_RV": 1, "global_btc_svi": 0, ...})
    df       = generate_scenario_series(hmm_results, scenario, T=252)
    mc, vars = run_monte_carlo(hmm_results, scenario, T=252, N=1000)
    plot_histogram_comparison(hmm_results, mc, vars)
    plot_regime_series(hmm_results, scenario)

[변환 설정]  ← run_pipeline.py 와 동일
  Global_RV        : log   → 역변환 exp
  global_btc_svi   : log1p → 역변환 expm1
  domestic_btc_svi : log1p → 역변환 expm1
  btc_volume_btc   : log1p → 역변환 expm1
  VKOSPI_resid     : None  → 역변환 없음
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm as hmmlearn_hmm

if TYPE_CHECKING:
    from preprocessing.gaussian_hmm import HMMComparison


# ══════════════════════════════════════════════
# 변수별 변환 설정 (run_pipeline.py 와 일치)
# ══════════════════════════════════════════════

VARIABLE_CONFIG: dict[str, dict] = {
    "Global_RV":         {"transform": "log",   "inverse": "exp"},
    "global_btc_svi":    {"transform": "log1p", "inverse": "expm1"},
    "domestic_btc_svi":  {"transform": "log1p", "inverse": "expm1"},
    "btc_volume_btc":    {"transform": "log1p", "inverse": "expm1"},
    "VKOSPI_resid":      {"transform": None,    "inverse": None},
}

# 시나리오 타입: {변수명 → 레짐 번호(μ 오름차순 기준)}
Scenario = Dict[str, int]

# final_scenarios_latest.csv 의 display 컬럼 → ScenarioHMMResult var_name 매핑
_DISPLAY_TO_VAR: dict[str, str] = {
    "Bitcoin_RV": "Global_RV",
    "VKOSPI":     "VKOSPI_resid",
    "KR_Volume":  "btc_volume_btc",
    "KR_SVI":     "domestic_btc_svi",
    "Global_SVI": "global_btc_svi",
}

_REGIME_COLORS = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]


# ══════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════

@dataclass
class ScenarioHMMResult:
    """단변량 Gaussian HMM 학습 결과 (시나리오 생성용).

    Attributes
    ----------
    mu, sigma     : 레짐별 emission 파라미터 (변환 공간, μ 오름차순)
    states        : Viterbi decoded 레짐 시퀀스 (T,)
    y_transformed : 변환 후 시계열 값 — 시각화·검증용
    series_index  : 원본 DatetimeIndex — 시계열 플롯용
    """
    var_name:      str
    n_states:      int
    transform:     Optional[str]           # 'log' | 'log1p' | None
    inverse:       Optional[str]           # 'exp' | 'expm1' | None
    mu:            np.ndarray              # (K,)
    sigma:         np.ndarray              # (K,)
    states:        np.ndarray              # (T,)
    y_transformed: Optional[np.ndarray] = None   # (T,)
    series_index:  Optional[pd.Index]   = None

    def print_summary(self) -> None:
        """레짐 파라미터 요약 출력."""
        print(f"\n  [{self.var_name}]  K={self.n_states}  "
              f"transform={self.transform}  inverse={self.inverse}")
        T = len(self.states)
        print(f"  {'Regime':<8} {'μ':>10} {'σ':>10} {'Occupancy':>12}")
        print(f"  {'-'*44}")
        for k in range(self.n_states):
            occ = (self.states == k).sum() / T
            print(f"  {k:<8} {self.mu[k]:>10.4f} {self.sigma[k]:>10.4f} {occ:>11.1%}")


# ══════════════════════════════════════════════
# 변환 유틸리티
# ══════════════════════════════════════════════

def apply_transform(y: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """시계열에 로그 변환을 적용한다.

    Parameters
    ----------
    y         : 원본 스케일 배열
    transform : 'log' | 'log1p' | None
    """
    if transform == "log":
        return np.log(y)
    if transform == "log1p":
        return np.log1p(y)
    return y.copy()


def apply_inverse(y: np.ndarray, inverse: Optional[str]) -> np.ndarray:
    """변환 공간의 값을 원본 스케일로 역변환한다.

    Parameters
    ----------
    y       : 변환 공간 배열
    inverse : 'exp' | 'expm1' | None
    """
    if inverse == "exp":
        return np.exp(y)
    if inverse == "expm1":
        return np.expm1(y)
    return y.copy()


# ══════════════════════════════════════════════
# 1. HMM 학습 (독립 실행용)
# ══════════════════════════════════════════════

def fit_scenario_hmm(
    series:    pd.Series,
    n_states:  int,
    transform: Optional[str] = "auto",
    n_init:    int   = 10,
    max_iter:  int   = 300,
    tol:       float = 1e-4,
    seed:      int   = 42,
) -> ScenarioHMMResult:
    """단변량 시계열에 Gaussian HMM을 학습한다.

    Parameters
    ----------
    series    : 원본 스케일 시계열 (pd.Series, DatetimeIndex 권장)
    n_states  : 레짐 수 K
    transform : 'auto' → VARIABLE_CONFIG 에서 series.name 으로 자동 조회.
                'log' | 'log1p' | None 으로 직접 지정 가능.
    n_init    : 다중 초기값 반복 횟수 (local optimum 회피)
    max_iter  : EM 최대 반복 수
    tol       : 수렴 기준 (log-likelihood 변화량)
    seed      : random seed (재현성)

    Returns
    -------
    ScenarioHMMResult
    """
    var_name = series.name or "unknown"
    cfg      = VARIABLE_CONFIG.get(var_name, {})

    if transform == "auto":
        transform = cfg.get("transform", None)

    inverse = cfg.get("inverse", None)
    if inverse is None:
        inverse = {"log": "exp", "log1p": "expm1"}.get(transform)  # type: ignore[arg-type]

    s_clean = series.dropna()
    y       = apply_transform(s_clean.values.astype(float), transform)
    obs     = y.reshape(-1, 1)

    best_ll    = -np.inf
    best_model = None

    for i in range(n_init):
        model = hmmlearn_hmm.GaussianHMM(
            n_components    = n_states,
            covariance_type = "diag",
            n_iter          = max_iter,
            tol             = tol,
            random_state    = seed + i,
            init_params     = "stmc",
            params          = "stmc",
            verbose         = False,
        )
        try:
            model.fit(obs)
            ll = model.score(obs)
            if ll > best_ll:
                best_ll    = ll
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError(f"HMM 학습 실패: '{var_name}' (K={n_states})")

    mu     = best_model.means_.flatten()
    sigma  = np.sqrt(best_model.covars_.flatten())
    states = best_model.predict(obs)

    # μ 오름차순 레짐 정렬
    order        = np.argsort(mu)
    inv_order    = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    mu     = mu[order]
    sigma  = sigma[order]
    states = inv_order[states]

    return ScenarioHMMResult(
        var_name      = var_name,
        n_states      = n_states,
        transform     = transform,
        inverse       = inverse,
        mu            = mu,
        sigma         = sigma,
        states        = states,
        y_transformed = y,
        series_index  = s_clean.index,
    )


# ══════════════════════════════════════════════
# 기존 파이프라인 결과 재활용
# ══════════════════════════════════════════════

def from_hmm_comparison(
    comp:   HMMComparison,
    series: pd.Series,
) -> ScenarioHMMResult:
    """gaussian_hmm.HMMComparison → ScenarioHMMResult 변환.

    run_pipeline.py 의 HMM 결과를 재사용할 때 사용한다.
    HMM 재학습 없이 기존 파라미터(μ, σ, states)를 그대로 활용한다.

    Parameters
    ----------
    comp   : run_hmm_pipeline() 또는 compare_hmm() 반환값
    series : 원본 스케일 시계열 (transform 및 series_index 복원용)
             series.name 이 VARIABLE_CONFIG 키와 일치해야 함
    """
    var_name = series.name or "unknown"
    cfg      = VARIABLE_CONFIG.get(var_name, {})
    transform = cfg.get("transform", None)
    inverse   = cfg.get("inverse", None)

    # optimal K 의 HMMResult 선택
    r = {1: comp.result_k1, 2: comp.result_k2, 3: comp.result_k3}[comp.optimal_K]

    s_clean = series.dropna()
    y       = apply_transform(s_clean.values.astype(float), transform)

    # HAR 잔차 등 시리즈 길이 불일치 대응
    T_min = min(len(y), len(r.states))

    return ScenarioHMMResult(
        var_name      = var_name,
        n_states      = r.K,
        transform     = transform,
        inverse       = inverse,
        mu            = r.mu.copy(),
        sigma         = r.sigma.copy(),
        states        = r.states[:T_min],
        y_transformed = y[:T_min],
        series_index  = s_clean.index[:T_min],
    )


def from_pipeline_results(pipeline_output: dict) -> Dict[str, ScenarioHMMResult]:
    """run_pipeline.main() 반환값에서 ScenarioHMMResult 딕셔너리를 구성한다.

    Parameters
    ----------
    pipeline_output : run_pipeline.main() 반환 dict
        필수 키: 'df_daily', 'df_weekly', 'har_result', 'hmm_svi', 'hmm_vol'

    Returns
    -------
    {변수명 → ScenarioHMMResult}  (변수 5개)
    """
    df_daily   = pipeline_output["df_daily"]
    df_weekly  = pipeline_output["df_weekly"]
    har_result = pipeline_output["har_result"]
    hmm_svi    = pipeline_output["hmm_svi"]    # {col: HMMComparison}
    hmm_vol    = pipeline_output["hmm_vol"]    # {col: HMMComparison}

    results: Dict[str, ScenarioHMMResult] = {}

    # SVI / Volume (주별)
    for col, comp in hmm_svi.items():
        series      = df_weekly[col].dropna().copy()
        series.name = col
        results[col] = from_hmm_comparison(comp, series)

    # Global_RV (일별)
    rv      = df_daily["Global_RV"].dropna().copy()
    rv.name = "Global_RV"
    results["Global_RV"] = from_hmm_comparison(hmm_vol["Global_RV"], rv)

    # VKOSPI_resid (HAR 잔차, 일별)
    vkospi_idx = df_daily["VKOSPI"].dropna().index
    resid_z    = pd.Series(
        har_result.residuals_z,
        index=vkospi_idx[22:22 + len(har_result.residuals_z)],
        name="VKOSPI_resid",
    )
    results["VKOSPI_resid"] = from_hmm_comparison(hmm_vol["VKOSPI_resid"], resid_z)

    return results


# ══════════════════════════════════════════════
# 2. 시나리오 정의
# ══════════════════════════════════════════════

def define_scenario(regime_dict: Dict[str, int]) -> Scenario:
    """레짐 조합(시나리오)을 딕셔너리로 정의한다.

    레짐 번호는 μ 오름차순 기준 (0 = 가장 낮은 수준 레짐).

    Parameters
    ----------
    regime_dict : {변수명: 레짐 번호}

    Examples
    --------
    >>> scenario = define_scenario({
    ...     "Global_RV":         1,   # 고변동성 레짐
    ...     "global_btc_svi":    0,   # 저관심 레짐
    ...     "domestic_btc_svi":  0,
    ...     "btc_volume_btc":    1,   # 고거래량 레짐
    ...     "VKOSPI_resid":      1,
    ... })
    """
    return dict(regime_dict)


# ══════════════════════════════════════════════
# CSV 시나리오 로드
# ══════════════════════════════════════════════

def scenarios_from_csv(
    csv_path:    "Path | str",
    hmm_results: Dict[str, "ScenarioHMMResult"],
) -> Dict[str, Scenario]:
    """final_scenarios_latest.csv → {Scenario_ID: Scenario} 변환.

    의미적 레이블(Low/Mid/High/Normal/Extreme)을 μ 오름차순 기준
    정수 레짐 번호로 변환한다.

    Parameters
    ----------
    csv_path    : final_scenarios_latest.csv 경로
    hmm_results : from_pipeline_results() 반환값 (K 조회용)

    Returns
    -------
    {Scenario_ID → {var_name → int}}
    """
    _VKOSPI_MAP = {
        2: {"Normal": 0, "Extreme": 1},
        3: {"Low": 0, "Normal": 1, "Extreme": 2},
    }
    _DEFAULT_MAP = {
        2: {"Low": 0, "High": 1},
        3: {"Low": 0, "Mid": 1, "High": 2},
    }

    # 변수별 레이블 → 정수 매핑 (K는 hmm_results 에서 조회)
    label_to_state: dict[str, dict[str, int]] = {}
    for disp_col, var_name in _DISPLAY_TO_VAR.items():
        res = hmm_results.get(var_name)
        if res is None:
            continue
        K = res.n_states
        if var_name == "VKOSPI_resid":
            label_to_state[disp_col] = _VKOSPI_MAP.get(K, {})
        else:
            label_to_state[disp_col] = _DEFAULT_MAP.get(K, {})

    df = pd.read_csv(Path(csv_path))
    named: Dict[str, Scenario] = {}

    for _, row in df.iterrows():
        sid = str(row["Scenario_ID"])
        scenario: Scenario = {}
        ok = True
        for disp_col, var_name in _DISPLAY_TO_VAR.items():
            label = row.get(disp_col)
            if pd.isna(label) or str(label).strip() in ("", "-"):
                print(f"  [!] {sid}: {disp_col} 레이블 누락 — 건너뜀")
                ok = False
                break
            state = label_to_state.get(disp_col, {}).get(str(label))
            if state is None:
                print(f"  [!] {sid}: {disp_col}='{label}' 미인식 — 건너뜀")
                ok = False
                break
            scenario[var_name] = state
        if ok:
            named[sid] = scenario

    print(f"  로드 완료: {len(named)}개 시나리오  {list(named.keys())}")
    return named


# ══════════════════════════════════════════════
# 3. 시나리오 시계열 생성
# ══════════════════════════════════════════════

def generate_scenario_series(
    hmm_results: Dict[str, ScenarioHMMResult],
    scenario:    Scenario,
    T:           int,
    seed:        int  = 42,
    inverse:     bool = True,
) -> pd.DataFrame:
    """시나리오 레짐의 emission 분포에서 T 개 시계열을 독립적으로 생성한다.

    각 변수를 N(μ_k, σ_k) 에서 i.i.d. 샘플링 후 역변환을 적용한다.
    변수 간 독립성 가정: 마르코프 전이 없이 순수 emission 샘플링.

    Parameters
    ----------
    hmm_results : {변수명 → ScenarioHMMResult}
    scenario    : {변수명 → 레짐 번호}
    T           : 생성 시계열 길이
    seed        : random seed (재현성)
    inverse     : True → 원본 스케일 역변환 적용 (exp / expm1)
                  False → 변환 공간 값 그대로 반환 (히스토그램 비교 등에 활용)

    Returns
    -------
    pd.DataFrame  shape (T, n_vars)
    """
    _validate_scenario(hmm_results, scenario)
    rng  = np.random.default_rng(seed)
    data = {}

    for var_name, res in hmm_results.items():
        k       = scenario[var_name]
        samples = rng.normal(loc=res.mu[k], scale=res.sigma[k], size=T)
        if inverse:
            samples = apply_inverse(samples, res.inverse)
        data[var_name] = samples

    return pd.DataFrame(data)


# ══════════════════════════════════════════════
# 4. 역변환 (별도 적용이 필요한 경우)
# ══════════════════════════════════════════════

def inverse_transform_dataframe(
    df:          pd.DataFrame,
    hmm_results: Dict[str, ScenarioHMMResult],
) -> pd.DataFrame:
    """변환 공간 DataFrame 을 원본 스케일로 역변환한다.

    generate_scenario_series(inverse=False) 결과를 사후에 역변환할 때 사용.

    Parameters
    ----------
    df          : 변환 공간 DataFrame (열 이름 = 변수명)
    hmm_results : 역변환 설정 조회용

    Returns
    -------
    pd.DataFrame  원본 스케일
    """
    out = {}
    for col in df.columns:
        res    = hmm_results.get(col)
        inv    = res.inverse if res else None
        out[col] = apply_inverse(df[col].values, inv)
    return pd.DataFrame(out, index=df.index)


# ══════════════════════════════════════════════
# 5. Monte Carlo 반복 생성
# ══════════════════════════════════════════════

def run_monte_carlo(
    hmm_results: Dict[str, ScenarioHMMResult],
    scenario:    Scenario,
    T:           int,
    N:           int,
    seed:        int  = 42,
    inverse:     bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """동일 시나리오를 N 회 반복 생성하여 3차원 배열로 반환한다.

    Parameters
    ----------
    T       : 시뮬레이션 시계열 길이
    N       : Monte Carlo 반복 횟수
    seed    : 기본 random seed (i번째 시뮬레이션은 seed+i 사용)
    inverse : True → 원본 스케일 역변환

    Returns
    -------
    mc_array  : np.ndarray  shape (N, T, n_vars)
    var_names : list[str]   axis=2 에 대응하는 변수명 순서
    """
    _validate_scenario(hmm_results, scenario)
    var_names = list(hmm_results.keys())
    n_vars    = len(var_names)
    mc_array  = np.empty((N, T, n_vars), dtype=float)

    for i in range(N):
        df = generate_scenario_series(
            hmm_results, scenario, T=T, seed=seed + i, inverse=inverse
        )
        mc_array[i] = df[var_names].values

    return mc_array, var_names


# ══════════════════════════════════════════════
# 6. 검증용 시각화
# ══════════════════════════════════════════════

def plot_histogram_comparison(
    hmm_results: Dict[str, ScenarioHMMResult],
    mc_array:    np.ndarray,
    var_names:   List[str],
    n_cols:      int = 3,
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """원본 데이터와 생성된 시나리오의 히스토그램을 비교한다.

    변환 공간에서 비교하므로 mc_array 는 inverse=False 로 생성 권장.

    Parameters
    ----------
    hmm_results : 원본 변환 시계열(y_transformed) 포함된 HMM 결과
    mc_array    : run_monte_carlo() 반환 배열 shape (N, T, n_vars)
    var_names   : mc_array axis=2 와 일치하는 변수명 리스트
    """
    n_vars = len(var_names)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, var in enumerate(var_names):
        ax  = axes[i]
        res = hmm_results.get(var)

        if res is not None and res.y_transformed is not None:
            ax.hist(res.y_transformed, bins=40, density=True,
                    alpha=0.55, color="#4e79a7", label="Original")

        sim_flat = mc_array[:, :, i].flatten()
        ax.hist(sim_flat, bins=40, density=True,
                alpha=0.55, color="#f28e2b", label="Scenario (MC)")

        xform = res.transform if res else "N/A"
        ax.set_title(f"{var}  (transform={xform})", fontsize=9)
        ax.set_xlabel("Value (transformed space)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Original vs Scenario — Histogram Comparison (Transformed Space)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Scenario] Histogram saved → {save_path}")
    return fig


def plot_regime_series(
    hmm_results: Dict[str, ScenarioHMMResult],
    scenario:    Optional[Scenario] = None,
    n_cols:      int = 2,
    save_path:   Optional[str] = None,
) -> plt.Figure:
    """각 변수의 원본 시계열에 레짐 색상을 입혀 시각화한다.

    시나리오가 주어지면 해당 레짐 구간을 진하게 강조 표시한다.

    Parameters
    ----------
    scenario : 강조할 레짐 조합. None 이면 모든 레짐 균등 표시.
    """
    var_names = list(hmm_results.keys())
    n_vars    = len(var_names)
    n_rows    = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, var in enumerate(var_names):
        ax  = axes[i]
        res = hmm_results[var]

        if res.y_transformed is None or res.series_index is None:
            ax.text(0.5, 0.5, f"{var}\n(시계열 데이터 없음)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(var)
            continue

        y   = res.y_transformed
        idx = res.series_index[:len(y)]
        ax.plot(idx, y, color="#222222", linewidth=0.7, alpha=0.85, zorder=2)

        for k in range(res.n_states):
            mask     = res.states == k
            color    = _REGIME_COLORS[k % len(_REGIME_COLORS)]
            occ      = mask.sum() / len(mask)
            label    = f"R{k}  μ={res.mu[k]:.3f}  σ={res.sigma[k]:.3f}  ({occ:.1%})"
            selected = scenario is not None and scenario.get(var) == k
            ax.fill_between(
                idx, y.min(), y.max(), where=mask,
                alpha=0.45 if selected else 0.18,
                color=color, label=label, zorder=1,
            )

        suffix = f"  ▶ Scenario: R{scenario[var]}" if (scenario and var in scenario) else ""
        ax.set_title(f"{var}{suffix}", fontsize=9)
        ax.set_ylabel(f"Value ({res.transform or 'original'})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Regime-Colored Time Series", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Scenario] Regime series saved → {save_path}")
    return fig


# ══════════════════════════════════════════════
# 통합 파이프라인
# ══════════════════════════════════════════════

def run_scenario_pipeline(
    hmm_results:     Dict[str, ScenarioHMMResult],
    named_scenarios: Dict[str, Scenario],
    T:               int,
    N:               int  = 1000,
    seed:            int  = 42,
    plot:            bool = True,
    save_dir:        Optional[str] = None,
) -> Dict[str, dict]:
    """여러 시나리오에 대해 Monte Carlo 생성 + 시각화를 일괄 실행한다.

    Parameters
    ----------
    hmm_results     : {변수명 → ScenarioHMMResult}
    named_scenarios : {시나리오명 → Scenario}
    T               : 생성 시계열 길이
    N               : Monte Carlo 반복 횟수
    seed            : 기본 random seed
    plot            : 시각화 여부
    save_dir        : 결과 저장 디렉토리 (None 이면 저장 안 함)

    Returns
    -------
    {시나리오명 → {
        "mc_array"  : np.ndarray (N, T, n_vars)  ← 변환 공간
        "mc_inv"    : np.ndarray (N, T, n_vars)  ← 원본 스케일
        "var_names" : list[str],
        "scenario"  : Scenario,
    }}
    """
    from pathlib import Path

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, dict] = {}

    for sc_name, scenario in named_scenarios.items():
        print(f"\n{'═'*60}")
        print(f"  Scenario: {sc_name}")
        print(f"{'═'*60}")
        for var, k in scenario.items():
            res = hmm_results.get(var)
            if res:
                print(f"    {var:<24} → Regime {k}"
                      f"  (μ={res.mu[k]:.4f}, σ={res.sigma[k]:.4f})")

        # 변환 공간 MC (히스토그램 비교용)
        mc_array, var_names = run_monte_carlo(
            hmm_results, scenario, T=T, N=N, seed=seed, inverse=False,
        )
        # 원본 스케일 MC (실제 활용용)
        mc_inv, _ = run_monte_carlo(
            hmm_results, scenario, T=T, N=N, seed=seed, inverse=True,
        )

        # 수치 요약
        print(f"\n  Monte Carlo: N={N}, T={T}  |  shape={mc_array.shape}")
        print(f"  {'Variable':<24} {'MC μ':>9} {'MC σ':>9}"
              f"  {'Orig μ':>9} {'Orig σ':>9}")
        for vi, var in enumerate(var_names):
            res    = hmm_results[var]
            mc_m   = mc_array[:, :, vi].mean()
            mc_s   = mc_array[:, :, vi].std()
            orig_m = res.y_transformed.mean() if res.y_transformed is not None else float("nan")
            orig_s = res.y_transformed.std()  if res.y_transformed is not None else float("nan")
            print(f"  {var:<24} {mc_m:>9.4f} {mc_s:>9.4f}  {orig_m:>9.4f} {orig_s:>9.4f}")

        if plot:
            sp_h = str(Path(save_dir) / f"hist_{sc_name}.png")  if save_dir else None
            sp_r = str(Path(save_dir) / f"regime_{sc_name}.png") if save_dir else None
            plot_histogram_comparison(hmm_results, mc_array, var_names, save_path=sp_h)
            plot_regime_series(hmm_results, scenario, save_path=sp_r)
            plt.show()

        outputs[sc_name] = {
            "mc_array":  mc_array,
            "mc_inv":    mc_inv,
            "var_names": var_names,
            "scenario":  scenario,
        }

    return outputs


# ══════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════

def _validate_scenario(
    hmm_results: Dict[str, ScenarioHMMResult],
    scenario:    Scenario,
) -> None:
    """시나리오 유효성 검사: 변수 누락 및 레짐 범위 확인."""
    for var in hmm_results:
        if var not in scenario:
            raise KeyError(f"시나리오에 '{var}' 누락 — 모든 변수의 레짐을 지정해야 합니다")
    for var, k in scenario.items():
        if var not in hmm_results:
            raise KeyError(f"HMM 결과에 '{var}' 없음")
        n = hmm_results[var].n_states
        if not (0 <= k < n):
            raise ValueError(
                f"'{var}': 레짐 {k} 요청, 가능 범위 [0, {n - 1}]"
            )


# ══════════════════════════════════════════════
# 독립 실행 예시
# ══════════════════════════════════════════════

if __name__ == "__main__":
    """
    [사전 설치]
      pip install hmmlearn

    [실행]
      python -m preprocessing.scenario_generator
    """
    import sys

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from preprocessing.run_pipeline import main as run_pipeline

    # ── Step 1: 전처리 파이프라인 실행 ──────────────────────
    print("=== 전처리 파이프라인 실행 중 ===")
    pipeline_out = run_pipeline(hmm_n_init=10, hmm_B=1000, save_plots=False, force_refit=False)

    # ── Step 2: HMM 결과 → ScenarioHMMResult 변환 ───────────
    print("\n=== HMM 결과 요약 ===")
    hmm_results = from_pipeline_results(pipeline_out)
    for res in hmm_results.values():
        res.print_summary()

    # ── Step 3: 시나리오 로드 (final_scenarios_latest.csv) ──
    _SCENARIO_CSV = _ROOT / "results" / "scenario_selection" / "final_scenarios_latest.csv"
    if _SCENARIO_CSV.exists():
        print(f"\n=== 시나리오 CSV 로드: {_SCENARIO_CSV.name} ===")
        scenarios = scenarios_from_csv(_SCENARIO_CSV, hmm_results)
    else:
        print(f"\n[!] 시나리오 CSV 없음: {_SCENARIO_CSV}")
        print("    analysis/scenario_selection.py 를 먼저 실행하세요.")
        print("    fallback: 기본 bull/stress 시나리오 사용\n")
        scenarios = {
            "bull": define_scenario({
                "Global_RV":        0,
                "global_btc_svi":   1,
                "domestic_btc_svi": 1,
                "btc_volume_btc":   1,
                "VKOSPI_resid":     0,
            }),
            "stress": define_scenario({
                "Global_RV":        1,
                "global_btc_svi":   0,
                "domestic_btc_svi": 0,
                "btc_volume_btc":   0,
                "VKOSPI_resid":     1,
            }),
        }

    # ── Step 4: 시나리오 파이프라인 실행 ────────────────────
    outputs = run_scenario_pipeline(
        hmm_results     = hmm_results,
        named_scenarios = scenarios,
        T               = 252,
        N               = 1000,
        seed            = 42,
        plot            = True,
        save_dir        = str(_ROOT / "results" / "scenarios"),
    )

    # ── Step 5: 단일 시나리오 시계열 확인 (원본 스케일) ──────
    first_sid = next(iter(scenarios))
    df_single = generate_scenario_series(
        hmm_results, scenarios[first_sid], T=252, seed=0, inverse=True,
    )
    print(f"\n=== 단일 시나리오 샘플 [{first_sid}] (첫 5행, 원본 스케일) ===")
    print(df_single.head())
