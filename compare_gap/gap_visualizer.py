"""
괴리율 모델 시각화 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
from scipy import stats
from statsmodels.tsa.stattools import acf


def plot_gap_level_paths(actual_level, simulated_level, save_path=None, monte_carlo_paths=None):
    """
    괴리율 수준 경로 비교 시각화 (몬테카를로 시뮬레이션 지원)
    
    Args:
        actual_level: 실제 괴리율 수준 시계열 y_t
        simulated_level: 시뮬레이션된 괴리율 수준 시계열 (단일 경로 또는 대표 경로)
        save_path: 저장 경로 (None이면 표시만)
        monte_carlo_paths: 몬테카를로 시뮬레이션 경로들 (n_paths x T 배열, 선택)
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    dates = range(len(actual_level))
    
    # 실제 괴리율 수준
    ax.plot(dates, actual_level, label='실제 괴리율', linewidth=2.5, alpha=0.9, color='black', zorder=10)
    
    # 몬테카를로 시뮬레이션 경로들
    if monte_carlo_paths is not None and len(monte_carlo_paths) > 0:
        # numpy 배열로 변환
        if isinstance(monte_carlo_paths, list):
            monte_carlo_paths = np.array(monte_carlo_paths)
        n_paths = len(monte_carlo_paths)
        
        # 모든 경로를 반투명하게 그림
        for i, path in enumerate(monte_carlo_paths):
            ax.plot(dates, path, alpha=0.1, linewidth=0.5, color='blue', zorder=1)
        
        # 평균 경로
        mean_path = np.mean(monte_carlo_paths, axis=0)
        ax.plot(dates, mean_path, label=f'평균 경로 (n={n_paths})', 
                linewidth=2, alpha=0.8, color='blue', linestyle='-', zorder=5)
        
        # 중앙값 경로
        median_path = np.median(monte_carlo_paths, axis=0)
        ax.plot(dates, median_path, label='중앙값 경로', 
                linewidth=2, alpha=0.8, color='green', linestyle='--', zorder=5)
        
        # 5%, 95% 신뢰구간
        p5_path = np.percentile(monte_carlo_paths, 5, axis=0)
        p95_path = np.percentile(monte_carlo_paths, 95, axis=0)
        ax.fill_between(dates, p5_path, p95_path, alpha=0.2, color='blue', 
                        label='90% 신뢰구간', zorder=2)
        
        # 대표 경로 (단일 경로로 전달된 경우)
        if simulated_level is not None:
            ax.plot(dates, simulated_level, label='대표 경로', 
                    linewidth=1.5, alpha=0.7, color='red', linestyle=':', zorder=6)
    else:
        # 단일 경로만 있는 경우
        if simulated_level is not None:
            ax.plot(dates, simulated_level, label='시뮬레이션 괴리율', 
                    linewidth=1.5, alpha=0.8, linestyle='--', color='blue')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('일수', fontsize=12)
    ax.set_ylabel('괴리율 (etf_premium)', fontsize=12)
    ax.set_title('괴리율 수준 경로 비교 (몬테카를로 시뮬레이션)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 괴리율 수준 경로 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gap_changes_distribution(actual_changes, simulated_changes, save_path=None, monte_carlo_changes=None):
    """
    괴리율 변화량 분포 비교 시각화 (히스토그램 + Q-Q plot, 몬테카를로 시뮬레이션 지원)
    
    Args:
        actual_changes: 실제 변화량 z_t = Δy_t
        simulated_changes: 시뮬레이션된 변화량 (단일 경로 또는 대표 경로)
        save_path: 저장 경로 (None이면 표시만)
        monte_carlo_changes: 몬테카를로 시뮬레이션 변화량들 (n_paths x T 배열, 선택)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    actual_values = np.array(actual_changes)
    
    # 히스토그램
    ax1 = axes[0]
    ax1.hist(actual_values, bins=50, alpha=0.6, label='실제', density=True, color='black', edgecolor='black')
    
    # 몬테카를로 시뮬레이션 변화량들
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        # numpy 배열로 변환
        if isinstance(monte_carlo_changes, list):
            monte_carlo_changes = np.array(monte_carlo_changes)
        # 모든 시뮬레이션 변화량을 하나로 합침
        all_sim_changes = monte_carlo_changes.flatten()
        n_paths = len(monte_carlo_changes)
        
        ax1.hist(all_sim_changes, bins=50, alpha=0.5, label=f'시뮬레이션 (n={n_paths})', 
                density=True, color='blue', edgecolor='blue', linewidth=0.5)
        
        # 평균 분포 (각 시점별 평균)
        mean_changes = np.mean(monte_carlo_changes, axis=0)
        ax1.hist(mean_changes, bins=50, alpha=0.7, label='평균 경로', 
                density=True, color='green', edgecolor='green', linewidth=1.5, histtype='step')
    else:
        # 단일 경로만 있는 경우
        if simulated_changes is not None:
            sim_values = np.array(simulated_changes)
            ax1.hist(sim_values, bins=50, alpha=0.6, label='시뮬레이션', density=True, color='red')
    
    ax1.set_xlabel('괴리율 변화량 (Δy_t)', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('괴리율 변화량 분포 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax2 = axes[1]
    stats.probplot(actual_values, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_label('실제')
    ax2.get_lines()[0].set_color('black')
    ax2.get_lines()[0].set_alpha(0.8)
    ax2.get_lines()[0].set_linewidth(2)
    
    # 몬테카를로 시뮬레이션 Q-Q plot
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        all_sim_changes = monte_carlo_changes.flatten()
        stats.probplot(all_sim_changes, dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label(f'시뮬레이션 (n={n_paths})')
        ax2.get_lines()[2].set_color('blue')
        ax2.get_lines()[2].set_alpha(0.6)
    elif simulated_changes is not None:
        sim_values = np.array(simulated_changes)
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
        print(f"  - 괴리율 변화량 분포 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gap_changes_timeseries(actual_changes, simulated_changes, save_path=None, monte_carlo_changes=None):
    """
    괴리율 변화량 시계열 비교 시각화
    
    Args:
        actual_changes: 실제 변화량
        simulated_changes: 시뮬레이션된 변화량
        save_path: 저장 경로 (None이면 표시만)
        monte_carlo_changes: 몬테카를로 시뮬레이션 변화량들 (선택)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    dates = range(len(actual_changes))
    
    # 실제 변화량
    ax1 = axes[0]
    ax1.plot(dates, actual_changes, linewidth=0.8, alpha=0.7, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('괴리율 변화량', fontsize=12)
    ax1.set_title('실제 괴리율 변화량 시계열', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 시뮬레이션 변화량
    ax2 = axes[1]
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        # 몬테카를로 평균 사용
        if isinstance(monte_carlo_changes, list):
            monte_carlo_changes = np.array(monte_carlo_changes)
        mean_sim_changes = np.mean(monte_carlo_changes, axis=0)
        ax2.plot(dates, mean_sim_changes, linewidth=0.8, alpha=0.7, color='red', label='평균 경로')
        
        # 신뢰구간
        p5_changes = np.percentile(monte_carlo_changes, 5, axis=0)
        p95_changes = np.percentile(monte_carlo_changes, 95, axis=0)
        ax2.fill_between(dates, p5_changes, p95_changes, alpha=0.2, color='red', label='90% 신뢰구간')
        ax2.legend(fontsize=9)
    elif simulated_changes is not None:
        ax2.plot(dates, simulated_changes, linewidth=0.8, alpha=0.7, color='red')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('일수', fontsize=12)
    ax2.set_ylabel('괴리율 변화량', fontsize=12)
    ax2.set_title('시뮬레이션 괴리율 변화량 시계열', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 괴리율 변화량 시계열 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gap_volatility_clustering(actual_changes, simulated_changes, max_lag=20, save_path=None):
    """
    괴리율 변화량 변동성 군집 비교 시각화 (ACF of z^2)
    
    Args:
        actual_changes: 실제 변화량
        simulated_changes: 시뮬레이션된 변화량
        max_lag: 최대 lag
        save_path: 저장 경로 (None이면 표시만)
    """
    actual_squared = np.array(actual_changes) ** 2
    sim_squared = np.array(simulated_changes) ** 2
    
    actual_acf = acf(actual_squared, nlags=max_lag, fft=True)
    sim_acf = acf(sim_squared, nlags=max_lag, fft=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = range(max_lag + 1)
    ax.plot(lags, actual_acf, 'o-', label='실제', linewidth=2, markersize=6, color='blue')
    ax.plot(lags, sim_acf, 's-', label='시뮬레이션', linewidth=2, markersize=6, color='red', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title('괴리율 변화량 변동성 군집 비교 (ACF of z²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - 변동성 군집 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gap_tail_comparison(actual_changes, simulated_changes, quantiles=[0.95, 0.975, 0.99, 0.995], save_path=None):
    """
    괴리율 변화량 분포 꼬리 비교 시각화
    
    Args:
        actual_changes: 실제 변화량
        simulated_changes: 시뮬레이션된 변화량
        quantiles: 비교할 분위수 리스트
        save_path: 저장 경로 (None이면 표시만)
    """
    actual_values = np.abs(np.array(actual_changes))
    sim_values = np.abs(np.array(simulated_changes))
    
    actual_quantiles = [np.quantile(actual_values, q) for q in quantiles]
    sim_quantiles = [np.quantile(sim_values, q) for q in quantiles]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(quantiles))
    width = 0.35
    
    ax.bar(x_pos - width/2, actual_quantiles, width, label='실제', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, sim_quantiles, width, label='시뮬레이션', alpha=0.7, color='red')
    
    ax.set_xlabel('분위수', fontsize=12)
    ax.set_ylabel('|괴리율 변화량|', fontsize=12)
    ax.set_title('괴리율 변화량 분포 꼬리 비교', fontsize=14, fontweight='bold')
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


def create_all_gap_visualizations(actual_changes, simulated_changes,
                                  actual_level=None, simulated_level=None,
                                  output_dir=None, 
                                  monte_carlo_changes_paths=None,
                                  monte_carlo_level_paths=None):
    """
    모든 괴리율 모델 시각화 생성 (몬테카를로 시뮬레이션 지원)
    
    Args:
        actual_changes: 실제 변화량 시계열 z_t = Δy_t
        simulated_changes: 시뮬레이션된 변화량 (대표 경로)
        actual_level: 실제 수준 시계열 y_t (선택, OU 모델용)
        simulated_level: 시뮬레이션된 수준 (대표 경로, 선택, OU 모델용)
        output_dir: 출력 디렉토리 (None이면 표시만)
        monte_carlo_changes_paths: 몬테카를로 시뮬레이션 변화량 경로들 (n_paths x T 배열, 선택)
        monte_carlo_level_paths: 몬테카를로 시뮬레이션 수준 경로들 (n_paths x T 배열, 선택, OU 모델용)
    """
    print(f"\n시각화 생성 중...")
    
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # 1. 괴리율 수준 경로 비교 (OU 모델용, 수준 데이터가 있는 경우)
    if actual_level is not None and simulated_level is not None:
        level_path = output_path / 'gap_level_paths.png' if output_path else None
        plot_gap_level_paths(actual_level, simulated_level, save_path=level_path,
                            monte_carlo_paths=monte_carlo_level_paths)
    
    # 2. 괴리율 변화량 분포 비교
    changes_dist_path = output_path / 'gap_changes_distribution.png' if output_path else None
    plot_gap_changes_distribution(actual_changes, simulated_changes, save_path=changes_dist_path,
                                 monte_carlo_changes=monte_carlo_changes_paths)
    
    # 3. 괴리율 변화량 시계열 비교
    changes_timeseries_path = output_path / 'gap_changes_timeseries.png' if output_path else None
    plot_gap_changes_timeseries(actual_changes, simulated_changes, save_path=changes_timeseries_path,
                               monte_carlo_changes=monte_carlo_changes_paths)
    
    # 4. 변동성 군집 비교 (몬테카를로 평균 사용)
    volatility_path = output_path / 'gap_volatility_clustering.png' if output_path else None
    if monte_carlo_changes_paths is not None and len(monte_carlo_changes_paths) > 0:
        # 몬테카를로 평균 변화량 사용
        mean_sim_changes = np.mean(monte_carlo_changes_paths, axis=0)
        plot_gap_volatility_clustering(actual_changes, mean_sim_changes, save_path=volatility_path)
    else:
        plot_gap_volatility_clustering(actual_changes, simulated_changes, save_path=volatility_path)
    
    # 5. 꼬리 분포 비교 (몬테카를로 모든 변화량 사용)
    tail_path = output_path / 'gap_tail_comparison.png' if output_path else None
    if monte_carlo_changes_paths is not None and len(monte_carlo_changes_paths) > 0:
        # numpy 배열로 변환 후 flatten
        if isinstance(monte_carlo_changes_paths, list):
            monte_carlo_changes_paths = np.array(monte_carlo_changes_paths)
        all_sim_changes = monte_carlo_changes_paths.flatten()
        plot_gap_tail_comparison(actual_changes, all_sim_changes, save_path=tail_path)
    else:
        plot_gap_tail_comparison(actual_changes, simulated_changes, save_path=tail_path)
    
    print(f"시각화 완료!")
