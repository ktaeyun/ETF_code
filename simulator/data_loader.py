"""
NAV 시뮬레이터용 데이터 로더
NAV_true에서 로그수익률 계산
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import settings


def load_nav_data():
    """
    NAV_true 데이터 로드 및 로그수익률 계산
    
    Returns:
        tuple: (nav_series, returns_series)
            - nav_series: NAV_true 시계열
            - returns_series: 로그수익률 시계열 r_t = Δlog(nav_true_t)
    """
    y_true_path = settings.get_y_true_variables_path()
    
    print(f"NAV 데이터 로드 중...")
    print(f"  - {y_true_path}")
    
    y_true_df = pd.read_csv(y_true_path)
    
    # Date 컬럼을 인덱스로 설정
    y_true_df['Date'] = pd.to_datetime(y_true_df['Date'])
    y_true_df = y_true_df.set_index('Date').sort_index()
    
    # NAV_true 추출
    nav_series = y_true_df['nav_true']
    
    # 로그수익률 계산: r_t = Δlog(nav_true_t) = log(nav_true_t) - log(nav_true_{t-1})
    log_nav = np.log(nav_series)
    returns_series = log_nav.diff().dropna()
    
    print(f"\n데이터 로드 완료:")
    print(f"  - NAV 샘플 수: {len(nav_series)}")
    print(f"  - 수익률 샘플 수: {len(returns_series)}")
    print(f"  - 기간: {returns_series.index[0]} ~ {returns_series.index[-1]}")
    
    return nav_series, returns_series
