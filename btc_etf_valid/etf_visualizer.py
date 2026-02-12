"""
BTC ETF 결합 결과 시각화
NAV(Heston) × (1 + gap(Heston-SV)) → ETF 가격·수익률 비교
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy import stats
from statsmodels.tsa.stattools import acf


def plot_etf_paths(actual_etf, simulated_etf_median, save_path=None, monte_carlo_etf_paths=None):
    """
    ETF 가격 경로 비교 (실제 etf_true vs 시뮬레이션 몬테카를로)
    
    Args:
        actual_etf: 실제 ETF 가격 시계열 (길이 T+1)
        simulated_etf_median: 시뮬레이션 대표 경로 (중앙값, 길이 T+1)
        save_path: 저장 경로
        monte_carlo_etf_paths: 몬테카를로 ETF 경로들 (n_paths x (T+1))
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    T_plus_1 = len(actual_etf)
    dates = np.arange(T_plus_1)
    
    ax.plot(dates, actual_etf, label='실제 ETF (etf_true)', linewidth=2.5, alpha=0.9, color='black', zorder=10)
    
    if monte_carlo_etf_paths is not None and len(monte_carlo_etf_paths) > 0:
        mc = np.asarray(monte_carlo_etf_paths)
        n_paths = len(mc)
        for i in range(n_paths):
            ax.plot(dates, mc[i], alpha=0.08, linewidth=0.5, color='blue', zorder=1)
        mean_path = np.mean(mc, axis=0)
        ax.plot(dates, mean_path, label=f'평균 경로 (n={n_paths})', linewidth=2, alpha=0.8, color='blue', linestyle='-', zorder=5)
        if simulated_etf_median is not None:
            ax.plot(dates, simulated_etf_median, label='중앙값 경로', linewidth=2, alpha=0.8, color='green', linestyle='--', zorder=5)
        p5 = np.percentile(mc, 5, axis=0)
        p95 = np.percentile(mc, 95, axis=0)
        ax.fill_between(dates, p5, p95, alpha=0.2, color='blue', label='90% 신뢰구간', zorder=2)
    elif simulated_etf_median is not None:
        ax.plot(dates, simulated_etf_median, label='시뮬 ETF (중앙값)', linewidth=1.5, alpha=0.8, linestyle='--', color='blue')
    
    ax.set_xlabel('일수', fontsize=12)
    ax.set_ylabel('ETF 가격', fontsize=12)
    ax.set_title('BTC ETF 가격 경로 (NAV×Heston + gap×Heston-SV 결합)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - ETF 경로 저장: {save_path}")
    plt.close()


def plot_etf_returns_distribution(actual_returns, simulated_median_returns, save_path=None, monte_carlo_returns=None):
    """
    ETF 수익률 분포 비교 (히스토그램 + Q-Q)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    actual_values = np.array(actual_returns)
    
    ax1 = axes[0]
    ax1.hist(actual_values, bins=50, alpha=0.6, label='실제 (etf_true)', density=True, color='black', edgecolor='black')
    
    if monte_carlo_returns is not None and len(monte_carlo_returns) > 0:
        mc = np.asarray(monte_carlo_returns)
        all_sim = mc.flatten()
        n_paths = len(mc)
        ax1.hist(all_sim, bins=50, alpha=0.5, label=f'시뮬 (n={n_paths})', density=True, color='blue', edgecolor='blue', linewidth=0.5)
        if simulated_median_returns is not None:
            ax1.hist(simulated_median_returns, bins=50, alpha=0.7, label='시뮬 중앙값 경로', density=True, color='green', 
                     edgecolor='green', linewidth=1.5, histtype='step')
    elif simulated_median_returns is not None:
        ax1.hist(np.array(simulated_median_returns), bins=50, alpha=0.6, label='시뮬', density=True, color='blue')
    
    ax1.set_xlabel('수익률', fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title('ETF 수익률 분포', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    stats.probplot(actual_values, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_label('실제')
    ax2.get_lines()[0].set_color('black')
    ax2.get_lines()[0].set_alpha(0.8)
    if monte_carlo_returns is not None and len(monte_carlo_returns) > 0:
        all_sim = np.asarray(monte_carlo_returns).flatten()
        stats.probplot(all_sim, dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label('시뮬')
        ax2.get_lines()[2].set_color('blue')
        ax2.get_lines()[2].set_alpha(0.6)
    ax2.set_title('Q-Q Plot (정규분포 기준)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - ETF 수익률 분포 저장: {save_path}")
    plt.close()


def plot_etf_returns_timeseries(actual_returns, simulated_median_returns, save_path=None):
    """ETF 수익률 시계열 비교"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    dates = np.arange(len(actual_returns))
    
    axes[0].plot(dates, actual_returns, linewidth=0.8, alpha=0.7, color='blue')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel('수익률', fontsize=12)
    axes[0].set_title('실제 ETF 수익률 시계열', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    if simulated_median_returns is not None:
        axes[1].plot(dates, simulated_median_returns, linewidth=0.8, alpha=0.7, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('일수', fontsize=12)
    axes[1].set_ylabel('수익률', fontsize=12)
    axes[1].set_title('시뮬 ETF 수익률 시계열 (중앙값 경로)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - ETF 수익률 시계열 저장: {save_path}")
    plt.close()


def plot_etf_volatility_clustering(actual_returns, simulated_median_returns, max_lag=20, save_path=None):
    """변동성 군집 (ACF of r²) 비교"""
    actual_sq = np.array(actual_returns) ** 2
    sim_sq = np.array(simulated_median_returns) ** 2 if simulated_median_returns is not None else None
    actual_acf = acf(actual_sq, nlags=max_lag, fft=True)
    sim_acf = acf(sim_sq, nlags=max_lag, fft=True) if sim_sq is not None else None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(max_lag + 1)
    ax.plot(lags, actual_acf, 'o-', label='실제', linewidth=2, markersize=6, color='blue')
    if sim_acf is not None:
        ax.plot(lags, sim_acf, 's-', label='시뮬 (중앙값)', linewidth=2, markersize=6, color='red', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title('ETF 수익률 변동성 군집 (ACF of r²)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - ETF 변동성 군집 저장: {save_path}")
    plt.close()


def plot_etf_tail_comparison(actual_returns, simulated_returns_flat, quantiles=(0.95, 0.975, 0.99, 0.995), save_path=None):
    """ETF 수익률 꼬리 분포 비교"""
    actual_abs = np.abs(np.array(actual_returns))
    sim_abs = np.abs(np.array(simulated_returns_flat))
    if len(sim_abs) == 0:
        return
    q_actual = [np.quantile(actual_abs, q) for q in quantiles]
    q_sim = [np.quantile(sim_abs, q) for q in quantiles]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(quantiles))
    w = 0.35
    ax.bar(x_pos - w/2, q_actual, w, label='실제', alpha=0.7, color='blue')
    ax.bar(x_pos + w/2, q_sim, w, label='시뮬', alpha=0.7, color='red')
    ax.set_xlabel('분위수', fontsize=12)
    ax.set_ylabel('|수익률|', fontsize=12)
    ax.set_title('ETF 수익률 꼬리 비교', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q*100}%' for q in quantiles])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - ETF 꼬리 비교 저장: {save_path}")
    plt.close()


def create_all_etf_visualizations(
    actual_etf,
    actual_etf_returns,
    simulated_etf_median,
    simulated_etf_returns_median,
    monte_carlo_etf_paths=None,
    monte_carlo_etf_returns=None,
    output_dir=None,
):
    """
    결합 BTC ETF 시각화 일괄 생성
    
    Args:
        actual_etf: 실제 ETF 가격 (etf_true, 길이 T+1)
        actual_etf_returns: 실제 ETF 수익률 (길이 T)
        simulated_etf_median: 시뮬 ETF 가격 중앙값 경로 (길이 T+1)
        simulated_etf_returns_median: 시뮬 ETF 수익률 중앙값 (길이 T)
        monte_carlo_etf_paths: 몬테카를로 ETF 경로 (n x (T+1))
        monte_carlo_etf_returns: 몬테카를로 ETF 수익률 (n x T)
        output_dir: 저장 디렉토리
    """
    from pathlib import Path
    if output_dir is None:
        out = Path(__file__).resolve().parent.parent / 'results' / 'btc_etf_valid' / 'plots'
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("\n[시각화] 결합 BTC ETF 시각화 생성")
    print("-" * 60)
    
    plot_etf_paths(
        actual_etf,
        simulated_etf_median,
        save_path=out / 'etf_paths.png',
        monte_carlo_etf_paths=monte_carlo_etf_paths,
    )
    plot_etf_returns_distribution(
        actual_etf_returns,
        simulated_etf_returns_median,
        save_path=out / 'etf_returns_distribution.png',
        monte_carlo_returns=monte_carlo_etf_returns,
    )
    plot_etf_returns_timeseries(
        actual_etf_returns,
        simulated_etf_returns_median,
        save_path=out / 'etf_returns_timeseries.png',
    )
    plot_etf_volatility_clustering(
        actual_etf_returns,
        simulated_etf_returns_median,
        save_path=out / 'etf_volatility_clustering.png',
    )
    if monte_carlo_etf_returns is not None and monte_carlo_etf_returns.size > 0:
        flat_sim = np.asarray(monte_carlo_etf_returns).flatten()
        plot_etf_tail_comparison(
            actual_etf_returns,
            flat_sim,
            save_path=out / 'etf_tail_comparison.png',
        )
    print(f"  시각화 저장 경로: {out}")
