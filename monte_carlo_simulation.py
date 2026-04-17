import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  MONTE CARLO SIMULATION
#  Finance Portfolio Project #5
# ============================================================

np.random.seed(42)

# ── CONFIG ──
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
WEIGHTS  = None          # None = equal weight, or e.g. [0.3, 0.2, 0.2, 0.2, 0.1]
PERIOD   = '2y'
NUM_SIMULATIONS  = 1000
NUM_DAYS         = 252    # 1 trading year
INITIAL_PORTFOLIO = 100000  # $100,000

CONFIDENCE_95 = 0.05
CONFIDENCE_99 = 0.01

# ============================================================
# 1. FETCH DATA
# ============================================================

def fetch_data(tickers, period='2y'):
    print("Fetching stock data...")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)['Close']
    raw.columns = tickers
    raw.dropna(inplace=True)
    print(f"  {len(raw)} trading days loaded for {len(tickers)} stocks\n")
    return raw

# ============================================================
# 2. CALCULATE RETURNS & STATS
# ============================================================

def get_return_stats(prices):
    returns     = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix   = returns.cov()
    return returns, mean_returns, cov_matrix

# ============================================================
# 3. MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(mean_returns, cov_matrix, weights, n_sim, n_days, initial):
    """
    Simulates n_sim portfolio paths over n_days using
    Cholesky decomposition to preserve correlation between assets.
    """
    n_assets = len(mean_returns)

    if weights is None:
        weights = np.ones(n_assets) / n_assets
    weights = np.array(weights)

    # Cholesky decomposition — preserves asset correlations
    L = np.linalg.cholesky(cov_matrix)

    portfolio_simulations = np.zeros((n_days, n_sim))

    for sim in range(n_sim):
        # Generate correlated random returns
        z            = np.random.standard_normal((n_assets, n_days))
        corr_returns = mean_returns.values[:, None] + L @ z

        # Daily portfolio return
        daily_port_return = weights @ corr_returns

        # Cumulative portfolio value
        price_path = initial * np.cumprod(1 + daily_port_return)
        portfolio_simulations[:, sim] = price_path

    return portfolio_simulations

# ============================================================
# 4. RISK METRICS
# ============================================================

def calculate_risk_metrics(simulations, initial):
    final_values  = simulations[-1, :]
    final_returns = (final_values - initial) / initial * 100

    # VaR
    var_95 = np.percentile(final_returns, CONFIDENCE_95 * 100)
    var_99 = np.percentile(final_returns, CONFIDENCE_99 * 100)

    # CVaR (Expected Shortfall)
    cvar_95 = final_returns[final_returns <= var_95].mean()
    cvar_99 = final_returns[final_returns <= var_99].mean()

    # Summary
    mean_ret  = final_returns.mean()
    median_ret = np.median(final_returns)
    best_case  = final_returns.max()
    worst_case = final_returns.min()
    prob_profit = (final_returns > 0).mean() * 100

    return {
        'final_values':  final_values,
        'final_returns': final_returns,
        'var_95':        var_95,
        'var_99':        var_99,
        'cvar_95':       cvar_95,
        'cvar_99':       cvar_99,
        'mean_ret':      mean_ret,
        'median_ret':    median_ret,
        'best_case':     best_case,
        'worst_case':    worst_case,
        'prob_profit':   prob_profit,
    }

# ============================================================
# 5. EFFICIENT FRONTIER
# ============================================================

def compute_efficient_frontier(mean_returns, cov_matrix, n_portfolios=3000):
    """
    Randomly samples portfolio weights to trace the efficient frontier.
    """
    n = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    weights_list = []

    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))
        weights_list.append(w)
        port_return = np.sum(mean_returns * w) * 252
        port_vol    = np.sqrt(w @ (cov_matrix * 252) @ w)
        sharpe      = port_return / port_vol if port_vol > 0 else 0
        results[0, i] = port_vol    * 100
        results[1, i] = port_return * 100
        results[2, i] = sharpe

    return results, weights_list

# ============================================================
# 6. PRINT SUMMARY
# ============================================================

def print_summary(metrics, weights, tickers, initial):
    w = weights if weights is not None else np.ones(len(tickers)) / len(tickers)
    print("\n" + "="*55)
    print("   MONTE CARLO SIMULATION — RESULTS")
    print("="*55)
    print(f"\n  Initial Investment : ${initial:,.0f}")
    print(f"  Simulations        : {NUM_SIMULATIONS:,}")
    print(f"  Time Horizon       : {NUM_DAYS} trading days (1 year)")
    print(f"\n  Portfolio Weights:")
    for t, wt in zip(tickers, w):
        print(f"    {t:<6} {wt*100:.1f}%")
    print(f"\n  Expected Return    : {metrics['mean_ret']:+.2f}%")
    print(f"  Median Return      : {metrics['median_ret']:+.2f}%")
    print(f"  Best Case          : {metrics['best_case']:+.2f}%")
    print(f"  Worst Case         : {metrics['worst_case']:+.2f}%")
    print(f"  Prob. of Profit    : {metrics['prob_profit']:.1f}%")
    print(f"\n  VaR  (95%)         : {metrics['var_95']:+.2f}%  (${initial * metrics['var_95']/100:,.0f})")
    print(f"  VaR  (99%)         : {metrics['var_99']:+.2f}%  (${initial * metrics['var_99']/100:,.0f})")
    print(f"  CVaR (95%)         : {metrics['cvar_95']:+.2f}%  (${initial * metrics['cvar_95']/100:,.0f})")
    print(f"  CVaR (99%)         : {metrics['cvar_99']:+.2f}%  (${initial * metrics['cvar_99']/100:,.0f})")
    print("="*55)

# ============================================================
# 7. DASHBOARD
# ============================================================

def plot_dashboard(simulations, metrics, ef_results, tickers, weights, initial, prices):

    C = {
        'bg':     '#0d1117',
        'panel':  '#161b22',
        'grid':   '#21262d',
        'text':   '#e6edf3',
        'green':  '#3fb950',
        'red':    '#f85149',
        'blue':   '#58a6ff',
        'purple': '#bc8cff',
        'yellow': '#d29922',
        'orange': '#f7931a',
        'teal':   '#39d0d8',
    }

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(C['bg'])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['text'], labelsize=8)
        ax.set_title(title, color=C['text'], fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=C['grid'], linewidth=0.5, linestyle='--')
        for sp in ax.spines.values():
            sp.set_color(C['grid'])

    # ── Panel 1: Monte Carlo Paths (full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, f'Monte Carlo Simulation — {NUM_SIMULATIONS:,} Portfolio Paths  |  {", ".join(tickers)}')
    x = np.arange(NUM_DAYS)

    # Plot a sample of paths
    for i in range(min(200, NUM_SIMULATIONS)):
        ax1.plot(x, simulations[:, i], alpha=0.03, color=C['blue'], lw=0.8)

    # Percentile bands
    p5   = np.percentile(simulations, 5,  axis=1)
    p25  = np.percentile(simulations, 25, axis=1)
    p50  = np.percentile(simulations, 50, axis=1)
    p75  = np.percentile(simulations, 75, axis=1)
    p95  = np.percentile(simulations, 95, axis=1)

    ax1.fill_between(x, p5,  p95, alpha=0.10, color=C['blue'],  label='5–95th pct')
    ax1.fill_between(x, p25, p75, alpha=0.20, color=C['blue'],  label='25–75th pct')
    ax1.plot(x, p50,  color=C['orange'], lw=2.0, label='Median path')
    ax1.plot(x, p5,   color=C['red'],    lw=1.0, ls='--', label='5th pct')
    ax1.plot(x, p95,  color=C['green'],  lw=1.0, ls='--', label='95th pct')
    ax1.axhline(initial, color=C['text'], lw=0.8, ls=':', alpha=0.5, label=f'Initial ${initial:,.0f}')

    ax1.set_xlabel('Trading Days', color=C['text'], fontsize=9)
    ax1.set_ylabel('Portfolio Value ($)', color=C['text'], fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=7, facecolor=C['panel'], labelcolor=C['text'], ncol=3)

    # ── Panel 2: Final Return Distribution ──
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, 'Final Return Distribution — 1 Year')
    ret = metrics['final_returns']
    ax2.hist(ret, bins=60, color=C['blue'], alpha=0.7, edgecolor='none', density=True)
    ax2.axvline(metrics['var_95'],  color=C['orange'], lw=1.5, ls='--', label=f"VaR 95%: {metrics['var_95']:.1f}%")
    ax2.axvline(metrics['var_99'],  color=C['red'],    lw=1.5, ls='--', label=f"VaR 99%: {metrics['var_99']:.1f}%")
    ax2.axvline(metrics['mean_ret'],color=C['green'],  lw=1.5, ls='-',  label=f"Mean: {metrics['mean_ret']:.1f}%")
    ax2.axvline(0, color=C['text'], lw=0.8, ls=':', alpha=0.5)
    ax2.set_xlabel('Return (%)', color=C['text'], fontsize=9)
    ax2.set_ylabel('Density',   color=C['text'], fontsize=9)
    ax2.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 3: Efficient Frontier ──
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, 'Efficient Frontier — Risk vs Return')
    sc = ax3.scatter(
        ef_results[0], ef_results[1],
        c=ef_results[2], cmap='RdYlGn',
        alpha=0.5, s=4
    )
    plt.colorbar(sc, ax=ax3, label='Sharpe Ratio').ax.yaxis.set_tick_params(colors=C['text'])

    # Mark equal weight portfolio
    w_eq  = np.ones(len(tickers)) / len(tickers)
    ret_eq = np.sum(prices.pct_change().dropna().mean() * w_eq) * 252 * 100
    vol_eq = np.sqrt(w_eq @ (prices.pct_change().dropna().cov() * 252) @ w_eq) * 100
    ax3.scatter(vol_eq, ret_eq, color=C['orange'], s=80, zorder=5, label='Equal Weight', marker='*')

    ax3.set_xlabel('Annual Volatility (%)', color=C['text'], fontsize=9)
    ax3.set_ylabel('Annual Return (%)',     color=C['text'], fontsize=9)
    ax3.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 4: VaR Summary Bar Chart ──
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, 'Risk Metrics — VaR & CVaR')
    labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
    values = [
        abs(metrics['var_95']),
        abs(metrics['var_99']),
        abs(metrics['cvar_95']),
        abs(metrics['cvar_99']),
    ]
    dollar_vals = [initial * v / 100 for v in values]
    bar_colors  = [C['orange'], C['red'], C['purple'], C['red']]
    bars = ax4.bar(labels, values, color=bar_colors, alpha=0.8)
    for bar, val, dval in zip(bars, values, dollar_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%\n${dval:,.0f}',
                 ha='center', va='bottom', color=C['text'], fontsize=7.5)
    ax4.set_ylabel('Loss (%)', color=C['text'], fontsize=9)
    ax4.set_ylim(0, max(values) * 1.4)

    # ── Panel 5: Normalized Stock Performance ──
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5, 'Historical Stock Performance — Normalized to 100')
    palette = [C['orange'], C['blue'], C['green'], C['purple'], C['yellow']]
    for ticker, color in zip(tickers, palette):
        normed = (prices[ticker] / prices[ticker].iloc[0]) * 100
        ax5.plot(prices.index, normed.values, label=ticker, color=color, lw=1.2)
    ax5.axhline(100, color=C['text'], lw=0.5, ls='--', alpha=0.4)
    ax5.set_ylabel('Indexed Return', color=C['text'], fontsize=9)
    ax5.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    fig.suptitle('MONTE CARLO SIMULATION  —  PORTFOLIO RISK ANALYSIS',
                 color=C['text'], fontsize=16, fontweight='bold', y=0.985)

    plt.savefig('monte_carlo_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("\nDashboard saved as monte_carlo_dashboard.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # 1. Fetch data
    prices = fetch_data(TICKERS, PERIOD)

    # 2. Return stats
    returns, mean_returns, cov_matrix = get_return_stats(prices)

    # 3. Set weights
    w = WEIGHTS if WEIGHTS is not None else np.ones(len(TICKERS)) / len(TICKERS)

    # 4. Run simulation
    print(f"Running {NUM_SIMULATIONS:,} simulations over {NUM_DAYS} days...")
    simulations = run_monte_carlo(mean_returns, cov_matrix, w,
                                  NUM_SIMULATIONS, NUM_DAYS, INITIAL_PORTFOLIO)
    print("  Done.\n")

    # 5. Risk metrics
    metrics = calculate_risk_metrics(simulations, INITIAL_PORTFOLIO)

    # 6. Efficient frontier
    print("Computing efficient frontier...")
    ef_results, _ = compute_efficient_frontier(mean_returns, cov_matrix)
    print("  Done.\n")

    # 7. Print summary
    print_summary(metrics, WEIGHTS, TICKERS, INITIAL_PORTFOLIO)

    # 8. Dashboard
    print("\nGenerating dashboard...")
    plot_dashboard(simulations, metrics, ef_results, TICKERS, w,
                   INITIAL_PORTFOLIO, prices)

    print("\nDone!")