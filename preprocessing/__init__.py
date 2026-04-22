"""
전처리 (Preprocessing) 패키지
===============================
외생변수 구간 분할을 위한 전처리 모듈 모음.

현재 모듈:
  - asvi_transformer : ASVI(Abnormal Search Volume Index) 변환
  - har_vkospi       : HAR(Heterogeneous Autoregressive) 모형 — VKOSPI 잔차 추출
  - bai_perron       : Bai-Perron (1998, 2003) 다중 구조 변화 검정
  - gaussian_hmm     : Gaussian HMM — K=2/3 상태 구간 분할 (EM + Bootstrap LRT)
"""

from preprocessing.asvi_transformer import (
    ASVI_COLUMN_MAP,
    DEFAULT_WINDOW,
    compute_asvi,
    transform_asvi,
    build_stats_table,
    plot_asvi_transformation,
    run_asvi_pipeline,
)
from preprocessing.har_vkospi import (
    LAG_D,
    LAG_W,
    LAG_M,
    HARResult,
    build_har_features,
    fit_har,
    plot_har_result,
    run_har_pipeline,
)
from preprocessing.bai_perron import (
    SUPF_CV,
    UDMAX_CV,
    WDMAX_CV,
    SEQ_CV,
    SegmentInfo,
    BaiPerronResult,
    fit_bai_perron,
    plot_bai_perron,
    run_bai_perron_pipeline,
)
from preprocessing.gaussian_hmm import (
    HMMResult,
    HMMComparison,
    fit_hmm,
    bootstrap_lrt,
    compare_hmm,
    plot_hmm_result,
    plot_hmm_comparison,
    run_hmm_pipeline,
)

__all__ = [
    # ASVI
    "ASVI_COLUMN_MAP",
    "DEFAULT_WINDOW",
    "compute_asvi",
    "transform_asvi",
    "build_stats_table",
    "plot_asvi_transformation",
    "run_asvi_pipeline",
    # HAR-VKOSPI
    "LAG_D",
    "LAG_W",
    "LAG_M",
    "HARResult",
    "build_har_features",
    "fit_har",
    "plot_har_result",
    "run_har_pipeline",
    # Bai-Perron
    "SUPF_CV",
    "UDMAX_CV",
    "WDMAX_CV",
    "SEQ_CV",
    "SegmentInfo",
    "BaiPerronResult",
    "fit_bai_perron",
    "plot_bai_perron",
    "run_bai_perron_pipeline",
    # Gaussian HMM
    "HMMResult",
    "HMMComparison",
    "fit_hmm",
    "bootstrap_lrt",
    "compare_hmm",
    "plot_hmm_result",
    "plot_hmm_comparison",
    "run_hmm_pipeline",
]
