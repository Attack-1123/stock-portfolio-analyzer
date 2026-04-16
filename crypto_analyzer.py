import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  CRYPTO MARKET ANALYZER
#  Finance Portfolio Project #3
# ============================================================

# --- CONFIG: change these as you like ---
CRYPTOS = {
    'Bitcoin':  'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Solana':   'SOL-USD',
    'BNB':      'BNB-USD',
    'Cardano':  'ADA-USD'
}
MAIN_CRYPTO = 'Bitcoin'   # Which coin gets the detailed chart
PERIOD = '1y'             # Data period: '1y', '6mo', '3mo'

# ============================================================
# 1. FETCH DATA
# ============================================================

def fetch_crypto_data(tickers, period='1y'):
    data = {}
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if not df.empty:
            data[name] = df
            print(f"  OK  {len(df)} days loaded")
        else:
            print(f"  FAILED to fetch {name}")
    return data

# ============================================================
# 2. TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(prices, period=14):
    """RSI tells us if a coin is overbought (>70) or oversold (<30)."""
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Bands that widen when volatile and narrow when calm."""
    sma   = prices.rolling(window=window).mean()
    std   = prices.rolling(window=window).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return upper, sma, lower

def add_indicators(df):
    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    df['MA20']         = close.rolling(20).mean()
    df['MA50']         = close.rolling(50).mean()
    df['RSI']          = calculate_rsi(close)
    df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger_bands(close)
    df['Daily_Return'] = close.pct_change()
    df['Volatility']   = df['Daily_Return'].rolling(30).std() * np.sqrt(365) * 100
    df['Volume_MA20']  = volume.rolling(20).mean()
    return df

# ============================================================
# 3. SUMMARY STATS
# ============================================================

def get_summary_stats(df, name):
    close = df['Close'].squeeze()
    ret   = df['Daily_Return'].dropna()

    cur   = close.iloc[-1]
    w1    = close.iloc[-7]  if len(close) >= 7  else close.iloc[0]
    m1    = close.iloc[-30] if len(close) >= 30 else close.iloc[0]
    y1    = close.iloc[0]

    ann_ret = ret.mean() * 365
    ann_vol = ret.std()  * np.sqrt(365)

    return {
        'Name':             name,
        'Current Price':    f"${cur:,.2f}",
        '1W Change':        f"{((cur/w1)-1)*100:+.2f}%",
        '1M Change':        f"{((cur/m1)-1)*100:+.2f}%",
        '1Y Change':        f"{((cur/y1)-1)*100:+.2f}%",
        'Ann. Volatility':  f"{ann_vol*100:.1f}%",
        'Sharpe Ratio':     f"{ann_ret/ann_vol:.2f}",
        'Current RSI':      f"{df['RSI'].iloc[-1]:.1f}",
        '52W High':         f"${close.max():,.2f}",
        '52W Low':          f"${close.min():,.2f}",
    }

# ============================================================
# 4. PRINT SUMMARY TABLE
# ============================================================

def print_summary(crypto_data):
    print("\n" + "="*60)
    print("   CRYPTO MARKET SUMMARY")
    print("="*60)
    for name, df in crypto_data.items():
        stats = get_summary_stats(df, name)
        print(f"\n  {stats['Name']}")
        print(f"  {'─'*38}")
        for k, v in stats.items():
            if k != 'Name':
                print(f"  {k:<22}  {v}")
    print("\n" + "="*60)

# ============================================================
# 5. CORRELATION MATRIX HELPER
# ============================================================

def build_correlation_matrix(crypto_data):
    closes = pd.DataFrame({
        name: df['Close'].squeeze()
        for name, df in crypto_data.items()
    })
    return closes.pct_change().dropna().corr()

# ============================================================
# 6. DASHBOARD
# ============================================================

def plot_dashboard(crypto_data, main_name='Bitcoin'):
    df     = crypto_data[main_name]
    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    # Dark theme colors
    C = {
        'bg':     '#0d1117',
        'panel':  '#161b22',
        'grid':   '#21262d',
        'text':   '#e6edf3',
        'orange': '#f7931a',
        'green':  '#3fb950',
        'red':    '#f85149',
        'blue':   '#58a6ff',
        'purple': '#bc8cff',
        'yellow': '#d29922',
    }

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(C['bg'])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['text'], labelsize=8)
        ax.set_title(title, color=C['text'], fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=C['grid'], linewidth=0.5, linestyle='--')
        for sp in ax.spines.values():
            sp.set_color(C['grid'])

    # ── Panel 1: Price + MAs + Bollinger Bands (full width) ──
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, f'{main_name}  —  Price · Moving Averages · Bollinger Bands')
    ax1.plot(close.index, close.values, color=C['orange'], lw=1.2, label='Price', zorder=3)
    ax1.plot(close.index, df['MA20'].values, color=C['blue'],   lw=1, ls='--', label='MA 20')
    ax1.plot(close.index, df['MA50'].values, color=C['purple'], lw=1, ls='--', label='MA 50')
    ax1.fill_between(close.index, df['BB_Upper'].values, df['BB_Lower'].values,
                     alpha=0.08, color=C['blue'])
    ax1.plot(close.index, df['BB_Upper'].values, color=C['blue'], lw=0.6, alpha=0.4)
    ax1.plot(close.index, df['BB_Lower'].values, color=C['blue'], lw=0.6, alpha=0.4)
    ax1.set_ylabel('Price (USD)', color=C['text'], fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=8, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 2: RSI ──
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, f'{main_name}  —  RSI (14-day)')
    rsi = df['RSI']
    ax2.plot(rsi.index, rsi.values, color=C['yellow'], lw=1.2)
    ax2.axhline(70, color=C['red'],   lw=0.8, ls='--', label='Overbought 70')
    ax2.axhline(30, color=C['green'], lw=0.8, ls='--', label='Oversold 30')
    ax2.fill_between(rsi.index, rsi.values, 70,
                     where=(rsi.values >= 70), alpha=0.15, color=C['red'])
    ax2.fill_between(rsi.index, rsi.values, 30,
                     where=(rsi.values <= 30), alpha=0.15, color=C['green'])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', color=C['text'], fontsize=9)
    ax2.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 3: Volume ──
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, f'{main_name}  —  Volume')
    bar_colors = [C['green'] if r >= 0 else C['red']
                  for r in df['Daily_Return'].fillna(0).values]
    ax3.bar(volume.index, volume.values, color=bar_colors, alpha=0.65, width=1)
    ax3.plot(volume.index, df['Volume_MA20'].values, color=C['yellow'], lw=1.2, label='20D Avg')
    ax3.set_ylabel('Volume', color=C['text'], fontsize=9)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    ax3.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 4: Correlation Heatmap ──
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, 'Correlation Matrix  —  1Y Daily Returns')
    corr   = build_correlation_matrix(crypto_data)
    names  = list(corr.columns)
    n      = len(names)
    im     = ax4.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(n)); ax4.set_yticks(range(n))
    ax4.set_xticklabels(names, rotation=30, ha='right', color=C['text'], fontsize=8)
    ax4.set_yticklabels(names, color=C['text'], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax4.text(j, i, f'{corr.values[i,j]:.2f}',
                     ha='center', va='center', fontsize=8, fontweight='bold', color='black')
    plt.colorbar(im, ax=ax4)

    # ── Panel 5: Normalized Performance ──
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5, 'Normalized Performance  —  Base = 100')
    palette = [C['orange'], C['blue'], C['green'], C['purple'], C['yellow']]
    for (name, d), color in zip(crypto_data.items(), palette):
        c   = d['Close'].squeeze()
        normed = (c / c.iloc[0]) * 100
        ax5.plot(c.index, normed.values, label=name, color=color, lw=1.2)
    ax5.axhline(100, color=C['text'], lw=0.5, ls='--', alpha=0.4)
    ax5.set_ylabel('Indexed Return', color=C['text'], fontsize=9)
    ax5.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])

    fig.suptitle('CRYPTO MARKET ANALYZER', color=C['text'],
                 fontsize=16, fontweight='bold', y=0.985)

    plt.savefig('crypto_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("\nDashboard saved as crypto_dashboard.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting Crypto Market Analyzer...\n")

    # Step 1: Download data
    crypto_data = fetch_crypto_data(CRYPTOS, period=PERIOD)

    # Step 2: Add indicators to each coin
    for name in crypto_data:
        crypto_data[name] = add_indicators(crypto_data[name])

    # Step 3: Print summary table in terminal
    print_summary(crypto_data)

    # Step 4: Generate dashboard
    print("\nGenerating dashboard...")
    plot_dashboard(crypto_data, main_name=MAIN_CRYPTO)

    print("\nAnalysis complete!")