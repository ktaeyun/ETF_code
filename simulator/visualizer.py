"""
NAV 시뮬레이터 시각화 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
from scipy import stats
from statsmodels.tsa.stattools import acf


def plot_nav_paths(actual_nav, simulated_nav, save_path=None):
    """
    NAV 경로 비교 시각화
    
    Args:
        actual_nav: 실제 NAV 시계열
        simulated_nav: 시뮬레이션된 NAV 시계열
        save_path: 저장 경로 (None이면 표시만)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = range(len(actual_nav))
    
    ax.plot(dates, actual_nav, label='실제 NAV', linewidth=1.5, alpha=0.8)
    ax.plot(dates, simulated_nav, label='시뮬레이션 NAV', linewidth=1.5, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('일수', fontsize=12)
    ax.set_ylabel('NAV', fontsize=12)
    ax.set_title('NAV 경로 비교', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - NAV 경로 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_returns_distribution(actual_returns, simulated_returns, save_path=None):
    """
    수익률 분포 비교 시각화 (히스토그램 + Q-Q plot)
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        save_path: 저장 경로 (None이면 표시만)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    actual_values = np.array(actual_returns)
    sim_values = np.array(simulated_returns)
    
    # 히스토그램
    ax1 = axes[0]
    ax1.hist(actual_values, bins=50, alpha=0.6, label='실제', density=True, color='blue')
    ax1.hist(sim_values, bins=50, alpha=0.6, label='시뮬레이션', density=True, color='red')
    ax1.set_xlabel('수익률', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('수익률 분포 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(actual_values, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_label('실제')
    ax2.get_lines()[0].set_color('blue')
    ax2.get_lines()[0].set_alpha(0.6)
    
    # 시뮬레이션 Q-Q plot 추가
    stats.probplot(sim_values, dist="norm", plot=ax2)
    ax2.get_lines()[2].set_label('시뮬레이션')
    ax2.get_lines()[2].set_color('red')
    ax2.get_lines()[2].set_alpha(0.6)
    
    ax2.set_title('Q-Q Plot (정규분포 기준)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 수익률 분포 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_volatility_clustering(actual_returns, simulated_returns, max_lag=20, save_path=None):
    """
    변동성 군집 비교 시각화 (ACF of r^2)
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        max_lag: 최대 lag
        save_path: 저장 경로 (None이면 표시만)
    """
    actual_squared = np.array(actual_returns) ** 2
    sim_squared = np.array(simulated_returns) ** 2
    
    actual_acf = acf(actual_squared, nlags=max_lag, fft=True)
    sim_acf = acf(sim_squared, nlags=max_lag, fft=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = range(max_lag + 1)
    ax.plot(lags, actual_acf, 'o-', label='실제', linewidth=2, markersize=6, color='blue')
    ax.plot(lags, sim_acf, 's-', label='시뮬레이션', linewidth=2, markersize=6, color='red', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title('변동성 군집 비교 (ACF of r²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 변동성 군집 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_jump_intervals(actual_jump_times, simulated_jump_times, save_path=None):
    """
    점프 간격 분포 비교 시각화
    
    Args:
        actual_jump_times: 실제 점프 시점
        simulated_jump_times: 시뮬레이션된 점프 시점
        save_path: 저장 경로 (None이면 표시만)
    """
    actual_intervals = np.diff(np.sort(actual_jump_times))
    sim_intervals = np.diff(np.sort(simulated_jump_times))
    
    if len(actual_intervals) == 0 or len(sim_intervals) == 0:
        print("  [경고] 점프 간격 데이터가 없어 시각화를 건너뜁니다.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 히스토그램
    ax1 = axes[0]
    ax1.hist(actual_intervals, bins=20, alpha=0.6, label='실제', density=True, color='blue')
    ax1.hist(sim_intervals, bins=20, alpha=0.6, label='시뮬레이션', density=True, color='red')
    
    # 지수분포 곡선 추가
    from scipy.stats import expon
    actual_lambda = 1 / np.mean(actual_intervals) if np.mean(actual_intervals) > 0 else 0
    sim_lambda = 1 / np.mean(sim_intervals) if np.mean(sim_intervals) > 0 else 0
    
    x = np.linspace(0, max(max(actual_intervals), max(sim_intervals)), 100)
    if actual_lambda > 0:
        ax1.plot(x, expon.pdf(x, scale=1/actual_lambda), 'b--', linewidth=2, label=f'실제 지수분포 (λ={actual_lambda:.4f})')
    if sim_lambda > 0:
        ax1.plot(x, expon.pdf(x, scale=1/sim_lambda), 'r--', linewidth=2, label=f'시뮬 지수분포 (λ={sim_lambda:.4f})')
    
    ax1.set_xlabel('점프 간격 (일)', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('점프 간격 분포', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot (지수분포 기준)
    ax2 = axes[1]
    if actual_lambda > 0:
        stats.probplot(actual_intervals, dist=expon, sparams=(0, 1/actual_lambda), plot=ax2)
        ax2.get_lines()[0].set_label('실제')
        ax2.get_lines()[0].set_color('blue')
        ax2.get_lines()[0].set_alpha(0.6)
    
    if sim_lambda > 0:
        stats.probplot(sim_intervals, dist=expon, sparams=(0, 1/sim_lambda), plot=ax2)
        ax2.get_lines()[2].set_label('시뮬레이션')
        ax2.get_lines()[2].set_color('red')
        ax2.get_lines()[2].set_alpha(0.6)
    
    ax2.set_title('Q-Q Plot (지수분포 기준)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 점프 간격 분포 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_returns_timeseries(actual_returns, simulated_returns, save_path=None):
    """
    수익률 시계열 비교 시각화
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        save_path: 저장 경로 (None이면 표시만)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    dates = range(len(actual_returns))
    
    # 실제 수익률
    ax1 = axes[0]
    ax1.plot(dates, actual_returns, linewidth=0.8, alpha=0.7, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('수익률', fontsize=12)
    ax1.set_title('실제 수익률 시계열', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 시뮬레이션 수익률
    ax2 = axes[1]
    ax2.plot(dates, simulated_returns, linewidth=0.8, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('일수', fontsize=12)
    ax2.set_ylabel('수익률', fontsize=12)
    ax2.set_title('시뮬레이션 수익률 시계열', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 수익률 시계열 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tail_comparison(actual_returns, simulated_returns, quantiles=[0.95, 0.975, 0.99, 0.995], save_path=None):
    """
    수익률 분포 꼬리 비교 시각화
    
    Args:
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        quantiles: 비교할 분위수 리스트
        save_path: 저장 경로 (None이면 표시만)
    """
    actual_values = np.abs(np.array(actual_returns))
    sim_values = np.abs(np.array(simulated_returns))
    
    actual_quantiles = [np.quantile(actual_values, q) for q in quantiles]
    sim_quantiles = [np.quantile(sim_values, q) for q in quantiles]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(quantiles))
    width = 0.35
    
    ax.bar(x_pos - width/2, actual_quantiles, width, label='실제', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, sim_quantiles, width, label='시뮬레이션', alpha=0.7, color='red')
    
    ax.set_xlabel('분위수', fontsize=12)
    ax.set_ylabel('|수익률|', fontsize=12)
    ax.set_title('수익률 분포 꼬리 비교', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q*100}%' for q in quantiles])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, (q_actual, q_sim) in enumerate(zip(actual_quantiles, sim_quantiles)):
        ax.text(i - width/2, q_actual, f'{q_actual:.4f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, q_sim, f'{q_sim:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 꼬리 분포 비교 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_visualizations(actual_nav, simulated_nav, actual_returns, simulated_returns,
                              actual_jump_times=None, simulated_jump_times=None,
                              output_dir=None):
    """
    모든 시각화 생성
    
    Args:
        actual_nav: 실제 NAV 시계열
        simulated_nav: 시뮬레이션된 NAV 시계열
        actual_returns: 실제 수익률
        simulated_returns: 시뮬레이션된 수익률
        actual_jump_times: 실제 점프 시점 (선택)
        simulated_jump_times: 시뮬레이션된 점프 시점 (선택)
        output_dir: 출력 디렉토리 (None이면 표시만)
    """
    print(f"\n시각화 생성 중...")
    
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # 1. NAV 경로 비교
    nav_path = output_path / 'nav_paths.png' if output_path else None
    plot_nav_paths(actual_nav, simulated_nav, save_path=nav_path)
    
    # 2. 수익률 분포 비교
    returns_dist_path = output_path / 'returns_distribution.png' if output_path else None
    plot_returns_distribution(actual_returns, simulated_returns, save_path=returns_dist_path)
    
    # 3. 변동성 군집 비교
    volatility_path = output_path / 'volatility_clustering.png' if output_path else None
    plot_volatility_clustering(actual_returns, simulated_returns, save_path=volatility_path)
    
    # 4. 수익률 시계열 비교
    timeseries_path = output_path / 'returns_timeseries.png' if output_path else None
    plot_returns_timeseries(actual_returns, simulated_returns, save_path=timeseries_path)
    
    # 5. 꼬리 분포 비교
    tail_path = output_path / 'tail_comparison.png' if output_path else None
    plot_tail_comparison(actual_returns, simulated_returns, save_path=tail_path)
    
    # 6. 점프 간격 분포 비교 (점프 데이터가 있는 경우)
    if actual_jump_times is not None and simulated_jump_times is not None:
        jump_intervals_path = output_path / 'jump_intervals.png' if output_path else None
        plot_jump_intervals(actual_jump_times, simulated_jump_times, save_path=jump_intervals_path)
    
    print(f"시각화 완료!")
