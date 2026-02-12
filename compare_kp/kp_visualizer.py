"""
김치 프리미엄(KP) 모델 시각화 모듈
gap_visualizer와 동일 구조, Kimchi Premium 라벨 사용
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy import stats
from statsmodels.tsa.stattools import acf


def plot_kp_level_paths(actual_level, simulated_level, save_path=None, monte_carlo_paths=None,
                        closest_path_idx=None):
    """김치 프리미엄(KP) 수준 경로 비교 시각화"""
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = range(len(actual_level))
    ax.plot(dates, actual_level, label='실제 김치 프리미엄', linewidth=2.5, alpha=0.9, color='black', zorder=10)

    if monte_carlo_paths is not None and len(monte_carlo_paths) > 0:
        if isinstance(monte_carlo_paths, list):
            monte_carlo_paths = np.array(monte_carlo_paths)
        n_paths = len(monte_carlo_paths)
        for i, path in enumerate(monte_carlo_paths):
            if closest_path_idx is not None and i == closest_path_idx:
                continue
            ax.plot(dates, path, alpha=0.1, linewidth=0.5, color='blue', zorder=1)
        if closest_path_idx is not None and 0 <= closest_path_idx < n_paths:
            ax.plot(dates, monte_carlo_paths[closest_path_idx],
                    label=f'실제와 가장 유사한 경로 (#{closest_path_idx + 1})',
                    linewidth=2.5, alpha=0.9, color='limegreen', linestyle='-', zorder=7)
        mean_path = np.mean(monte_carlo_paths, axis=0)
        ax.plot(dates, mean_path, label=f'평균 경로 (n={n_paths})', linewidth=2, alpha=0.8, color='blue', linestyle='-', zorder=5)
        median_path = np.median(monte_carlo_paths, axis=0)
        ax.plot(dates, median_path, label='중앙값 경로', linewidth=2, alpha=0.8, color='green', linestyle='--', zorder=5)
        p5 = np.percentile(monte_carlo_paths, 5, axis=0)
        p95 = np.percentile(monte_carlo_paths, 95, axis=0)
        ax.fill_between(dates, p5, p95, alpha=0.2, color='blue', label='90% 신뢰구간', zorder=2)
        if simulated_level is not None:
            ax.plot(dates, simulated_level, label='대표 경로(중앙값)', linewidth=1.5, alpha=0.7, color='red', linestyle=':', zorder=6)
    else:
        if simulated_level is not None:
            ax.plot(dates, simulated_level, label='시뮬레이션 KP', linewidth=1.5, alpha=0.8, linestyle='--', color='blue')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('일수', fontsize=12)
    ax.set_ylabel('김치 프리미엄 (Kimchi Premium)', fontsize=12)
    ax.set_title('김치 프리미엄 수준 경로 비교 (몬테카를로 시뮬레이션)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - KP 수준 경로 저장: {save_path}")
    plt.close()


def plot_kp_changes_distribution(actual_changes, simulated_changes, save_path=None, monte_carlo_changes=None):
    """KP 변화량 분포 비교 (히스토그램 + Q-Q)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    actual_values = np.array(actual_changes)
    ax1 = axes[0]
    ax1.hist(actual_values, bins=50, alpha=0.6, label='실제', density=True, color='black', edgecolor='black')
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        mc = np.array(monte_carlo_changes)
        n_paths = len(mc)
        ax1.hist(mc.flatten(), bins=50, alpha=0.5, label=f'시뮬 (n={n_paths})', density=True, color='blue', edgecolor='blue', linewidth=0.5)
        ax1.hist(np.mean(mc, axis=0), bins=50, alpha=0.7, label='평균 경로', density=True, color='green', edgecolor='green', linewidth=1.5, histtype='step')
    elif simulated_changes is not None:
        ax1.hist(np.array(simulated_changes), bins=50, alpha=0.6, label='시뮬', density=True, color='red')
    ax1.set_xlabel('KP 변화량 (Δy_t)', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('김치 프리미엄 변화량 분포', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    stats.probplot(actual_values, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_label('실제')
    ax2.get_lines()[0].set_color('black')
    ax2.get_lines()[0].set_alpha(0.8)
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        stats.probplot(np.array(monte_carlo_changes).flatten(), dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label('시뮬')
        ax2.get_lines()[2].set_color('blue')
        ax2.get_lines()[2].set_alpha(0.6)
    elif simulated_changes is not None:
        stats.probplot(np.array(simulated_changes), dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label('시뮬')
        ax2.get_lines()[2].set_color('red')
        ax2.get_lines()[2].set_alpha(0.6)
    ax2.set_title('Q-Q Plot (정규분포 기준)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - KP 변화량 분포 저장: {save_path}")
    plt.close()


def plot_kp_changes_timeseries(actual_changes, simulated_changes, save_path=None, monte_carlo_changes=None):
    """KP 변화량 시계열 비교"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    dates = range(len(actual_changes))
    axes[0].plot(dates, actual_changes, linewidth=0.8, alpha=0.7, color='blue')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('KP 변화량', fontsize=12)
    axes[0].set_title('실제 김치 프리미엄 변화량 시계열', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if monte_carlo_changes is not None and len(monte_carlo_changes) > 0:
        mc = np.array(monte_carlo_changes)
        axes[1].plot(dates, np.mean(mc, axis=0), linewidth=0.8, alpha=0.7, color='red', label='평균 경로')
        p5, p95 = np.percentile(mc, 5, axis=0), np.percentile(mc, 95, axis=0)
        axes[1].fill_between(dates, p5, p95, alpha=0.2, color='red', label='90% 신뢰구간')
        axes[1].legend(fontsize=9)
    elif simulated_changes is not None:
        axes[1].plot(dates, simulated_changes, linewidth=0.8, alpha=0.7, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('일수', fontsize=12)
    axes[1].set_ylabel('KP 변화량', fontsize=12)
    axes[1].set_title('시뮬레이션 김치 프리미엄 변화량 시계열', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - KP 변화량 시계열 저장: {save_path}")
    plt.close()


def plot_kp_volatility_clustering(actual_changes, simulated_changes, max_lag=20, save_path=None):
    """KP 변화량 변동성 군집 (ACF of z²)"""
    actual_sq = np.array(actual_changes) ** 2
    sim_sq = np.array(simulated_changes) ** 2
    actual_acf = acf(actual_sq, nlags=max_lag, fft=True)
    sim_acf = acf(sim_sq, nlags=max_lag, fft=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = range(max_lag + 1)
    ax.plot(lags, actual_acf, 'o-', label='실제', linewidth=2, markersize=6, color='blue')
    ax.plot(lags, sim_acf, 's-', label='시뮬', linewidth=2, markersize=6, color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title('김치 프리미엄 변화량 변동성 군집 (ACF of z²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - KP 변동성 군집 저장: {save_path}")
    plt.close()


def plot_kp_tail_comparison(actual_changes, simulated_changes, quantiles=[0.95, 0.975, 0.99, 0.995], save_path=None):
    """KP 변화량 꼬리 비교"""
    actual_abs = np.abs(np.array(actual_changes))
    sim_abs = np.abs(np.array(simulated_changes))
    q_actual = [np.quantile(actual_abs, q) for q in quantiles]
    q_sim = [np.quantile(sim_abs, q) for q in quantiles]
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(quantiles))
    w = 0.35
    ax.bar(x_pos - w/2, q_actual, w, label='실제', alpha=0.7, color='blue')
    ax.bar(x_pos + w/2, q_sim, w, label='시뮬', alpha=0.7, color='red')
    ax.set_xlabel('분위수', fontsize=12)
    ax.set_ylabel('|KP 변화량|', fontsize=12)
    ax.set_title('김치 프리미엄 변화량 꼬리 비교', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q*100}%' for q in quantiles])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - KP 꼬리 비교 저장: {save_path}")
    plt.close()


def create_all_kp_visualizations(actual_changes, simulated_changes,
                                  actual_level=None, simulated_level=None,
                                  output_dir=None,
                                  monte_carlo_changes_paths=None,
                                  monte_carlo_level_paths=None,
                                  closest_level_path_idx=None):
    """김치 프리미엄(KP) 모델 시각화 일괄 생성"""
    from pathlib import Path
    print("\n시각화 생성 중 (김치 프리미엄)...")
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    if actual_level is not None and simulated_level is not None:
        p = output_path / 'kp_level_paths.png' if output_path else None
        plot_kp_level_paths(actual_level, simulated_level, save_path=p,
                            monte_carlo_paths=monte_carlo_level_paths, closest_path_idx=closest_level_path_idx)
    p = output_path / 'kp_changes_distribution.png' if output_path else None
    plot_kp_changes_distribution(actual_changes, simulated_changes, save_path=p, monte_carlo_changes=monte_carlo_changes_paths)
    p = output_path / 'kp_changes_timeseries.png' if output_path else None
    plot_kp_changes_timeseries(actual_changes, simulated_changes, save_path=p, monte_carlo_changes=monte_carlo_changes_paths)
    p = output_path / 'kp_volatility_clustering.png' if output_path else None
    if monte_carlo_changes_paths is not None and len(monte_carlo_changes_paths) > 0:
        plot_kp_volatility_clustering(actual_changes, np.mean(monte_carlo_changes_paths, axis=0), save_path=p)
    else:
        plot_kp_volatility_clustering(actual_changes, simulated_changes, save_path=p)
    p = output_path / 'kp_tail_comparison.png' if output_path else None
    if monte_carlo_changes_paths is not None and len(monte_carlo_changes_paths) > 0:
        plot_kp_tail_comparison(actual_changes, np.array(monte_carlo_changes_paths).flatten(), save_path=p)
    else:
        plot_kp_tail_comparison(actual_changes, simulated_changes, save_path=p)
    print("시각화 완료!")
