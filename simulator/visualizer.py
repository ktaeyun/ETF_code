"""
NAV 시뮬레이터 시각화 모듈
compare와 동일한 형식: 실제 vs 시뮬레이션 NAV/수익률, 몬테카를로 경로·신뢰구간
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def _to_array(x):
    if x is None:
        return None
    return np.asarray(x).flatten()


_MODULE_META = {
    "nav":      {"label": "NAV",      "model": "ARIMA-GARCH-t",     "ylabel": "NAV"},
    "gap":      {"label": "GAP",      "model": "OU with exog",       "ylabel": "ETF 프리미엄"},
    "kp":       {"label": "KP",       "model": "Threshold-OU",       "ylabel": "김치 프리미엄"},
    "combined": {"label": "결합",     "model": "NAV×(1+GAP)",        "ylabel": "ETF 가격"},
}


def plot_nav_paths(actual_nav, simulated_nav, save_path=None, monte_carlo_nav_paths=None, module="nav"):
    """
    경로 비교 (실제 vs 시뮬레이션, 몬테카를로 팬)
    """
    meta = _MODULE_META.get(module, _MODULE_META["nav"])
    actual_nav = _to_array(actual_nav)
    simulated_nav = _to_array(simulated_nav)
    T = len(actual_nav)
    dates = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, actual_nav, label=f'실제 {meta["label"]}', linewidth=2.5, alpha=0.9, color='black', zorder=10)

    if monte_carlo_nav_paths is not None and len(monte_carlo_nav_paths) > 0:
        mc = np.asarray(monte_carlo_nav_paths)
        if mc.ndim == 1:
            mc = mc.reshape(1, -1)
        n_paths = len(mc)
        for i in range(n_paths):
            ax.plot(dates, mc[i], alpha=0.08, linewidth=0.5, color='blue', zorder=1)
        mean_path = np.mean(mc, axis=0)
        median_path = np.median(mc, axis=0)
        ax.plot(dates, mean_path, label=f'평균 경로 (n={n_paths})', linewidth=2, alpha=0.8, color='blue', linestyle='-', zorder=5)
        ax.plot(dates, median_path, label='중앙값 경로', linewidth=2, alpha=0.8, color='green', linestyle='--', zorder=5)
        p5 = np.percentile(mc, 5, axis=0)
        p95 = np.percentile(mc, 95, axis=0)
        ax.fill_between(dates, p5, p95, alpha=0.2, color='blue', label='90% 신뢰구간', zorder=2)
    if simulated_nav is not None and len(simulated_nav) == T:
        ax.plot(dates, simulated_nav, label='대표 경로(중앙값)', linewidth=1.5, alpha=0.7, color='red', linestyle=':', zorder=6)

    ax.set_xlabel('일수', fontsize=12)
    ax.set_ylabel(meta["ylabel"], fontsize=12)
    ax.set_title(f'{meta["label"]} 경로 비교 ({meta["model"]} 몬테카를로)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - {meta['label']} 경로 저장: {save_path}")
    plt.close()


def plot_returns_distribution(actual_returns, simulated_returns, save_path=None, monte_carlo_returns_paths=None, module="nav"):
    """
    변화량 분포 비교 (히스토그램 + Q-Q)
    """
    from scipy import stats
    meta = _MODULE_META.get(module, _MODULE_META["nav"])
    actual_values = _to_array(actual_returns)
    xlabel = "Log Return" if module == "nav" else f"{meta['label']} 변화량"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.hist(actual_values, bins=50, alpha=0.6, label='실제', density=True, color='black', edgecolor='black')

    if monte_carlo_returns_paths is not None and len(monte_carlo_returns_paths) > 0:
        mc = np.asarray(monte_carlo_returns_paths)
        if mc.ndim == 1:
            mc = mc.reshape(1, -1)
        all_sim = mc.flatten()
        n_paths = len(mc)
        ax1.hist(all_sim, bins=50, alpha=0.5, label=f'시뮬 (n={n_paths})', density=True, color='blue', edgecolor='blue', linewidth=0.5)
        if simulated_returns is not None:
            med = _to_array(simulated_returns)
            ax1.hist(med, bins=50, alpha=0.7, label='시뮬 중앙값 경로', density=True, color='green', edgecolor='green', linewidth=1.5, histtype='step')
    elif simulated_returns is not None:
        ax1.hist(_to_array(simulated_returns), bins=50, alpha=0.6, label='시뮬', density=True, color='blue')

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('밀도', fontsize=12)
    ax1.set_title(f'{meta["label"]} 분포 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    stats.probplot(actual_values, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_label('실제')
    ax2.get_lines()[0].set_color('black')
    ax2.get_lines()[0].set_alpha(0.8)
    if monte_carlo_returns_paths is not None and len(monte_carlo_returns_paths) > 0:
        all_sim = np.asarray(monte_carlo_returns_paths).flatten()
        stats.probplot(all_sim, dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label('시뮬')
        ax2.get_lines()[2].set_color('blue')
        ax2.get_lines()[2].set_alpha(0.6)
    elif simulated_returns is not None:
        stats.probplot(_to_array(simulated_returns), dist="norm", plot=ax2)
        ax2.get_lines()[2].set_label('시뮬')
        ax2.get_lines()[2].set_color('blue')
        ax2.get_lines()[2].set_alpha(0.6)
    ax2.set_title('Q-Q Plot (정규분포 기준)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - {meta['label']} 분포 저장: {save_path}")
    plt.close()


def plot_returns_timeseries(actual_returns, simulated_returns, save_path=None, module="nav"):
    """
    변화량 시계열 비교 (실제 / 시뮬레이션)
    """
    meta = _MODULE_META.get(module, _MODULE_META["nav"])
    actual_values = _to_array(actual_returns)
    simulated_values = _to_array(simulated_returns) if simulated_returns is not None else None
    T = len(actual_values)
    dates = np.arange(T)
    ylabel = "Log Return" if module == "nav" else f"{meta['label']} 변화량"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(dates, actual_values, linewidth=0.8, alpha=0.7, color='blue')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_title(f'실제 {meta["label"]} 변화량 시계열', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    if simulated_values is not None and len(simulated_values) == T:
        axes[1].plot(dates, simulated_values, linewidth=0.8, alpha=0.7, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('일수', fontsize=12)
    axes[1].set_ylabel(ylabel, fontsize=12)
    axes[1].set_title(f'시뮬레이션 {meta["label"]} 변화량 시계열 (대표 경로)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - {meta['label']} 변화량 시계열 저장: {save_path}")
    plt.close()


def create_all_visualizations(
    actual_nav,
    simulated_nav,
    actual_returns,
    simulated_returns,
    output_dir,
    monte_carlo_nav_paths=None,
    monte_carlo_returns_paths=None,
    module="nav",
):
    """
    경로, 분포, 시계열 플롯 저장.
    module: "nav" | "gap" | "kp" | "combined"
    """
    import os
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    plot_nav_paths(
        actual_nav, simulated_nav,
        save_path=os.path.join(output_dir, 'nav_paths.png'),
        monte_carlo_nav_paths=monte_carlo_nav_paths,
        module=module,
    )
    plot_returns_distribution(
        actual_returns, simulated_returns,
        save_path=os.path.join(output_dir, 'returns_distribution.png'),
        monte_carlo_returns_paths=monte_carlo_returns_paths,
        module=module,
    )
    plot_returns_timeseries(
        actual_returns, simulated_returns,
        save_path=os.path.join(output_dir, 'returns_timeseries.png'),
        module=module,
    )
