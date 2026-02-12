"""
NAV 시뮬레이터 패키지
ARIMAX-GARCH-t: Log Return을 종속변수, Hash Rate + Unique Addresses를 독립변수로 사용.
검증: 통계적 검정(PIT-KS, VaR-Kupiec, ES) + Weighted Multi-band Capture Rate(WMCR) 등.
"""

from simulator.data_loader import load_nav_exog_and_returns
from simulator.arima_garch_t_nav_simulator import (
    fit_arimax_garch_t,
    NavArimaGarchTSimulator,
    log_returns_to_nav,
)
from simulator.visualizer import create_all_visualizations
from simulator.wmcr_test import compute_wmcr, wmcr_pvalue_calibration, wmcr_binomial_test
from simulator.gap_ou_simulator import fit_gap_ou, GapOUSimulator
from simulator.kp_threshold_ou_simulator import fit_kp_threshold_ou, KPThresholdOUSimulator
from simulator.data_loader import load_gap_exog, load_kp_exog

__all__ = [
    "load_nav_exog_and_returns",
    "load_gap_exog",
    "load_kp_exog",
    "fit_arimax_garch_t",
    "NavArimaGarchTSimulator",
    "log_returns_to_nav",
    "fit_gap_ou",
    "GapOUSimulator",
    "fit_kp_threshold_ou",
    "KPThresholdOUSimulator",
    "create_all_visualizations",
    "compute_wmcr",
    "wmcr_pvalue_calibration",
    "wmcr_binomial_test",
]
