import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.request
import json
import re
from datetime import datetime, timedelta

# ============================================================
#  SENTIMENT TRADING SIGNAL
#  Finance Portfolio Project #6
# ============================================================

TICKERS = {
    'Apple':    'AAPL',
    'Microsoft':'MSFT',
    'Nvidia':   'NVDA',
    'Amazon':   'AMZN',
    'Tesla':    'TSLA',
}
MAIN_TICKER  = 'AAPL'
MAIN_NAME    = 'Apple'
PERIOD       = '6mo'

BUY_THRESHOLD  =  0.15
SELL_THRESHOLD = -0.15

# ============================================================
# 1. FETCH STOCK PRICES
# ============================================================

def fetch_prices(tickers, period='6mo'):
    print("Fetching stock prices...")
    symbols = list(tickers.values())
    data    = yf.download(symbols, period=period, auto_adjust=True, progress=False)['Close']
    if len(symbols) == 1:
        data = data.to_frame(name=symbols[0])
    else:
        data.columns = symbols
    data.dropna(inplace=True)
    print(f"  {len(data)} trading days loaded\n")
    return data

# ============================================================
# 2. FETCH NEWS HEADLINES
# ============================================================

def simulate_headlines(company_name, n=40):
    positive = [
        f"{company_name} beats earnings estimates by wide margin",
        f"{company_name} stock surges after strong quarterly results",
        f"Analysts upgrade {company_name} with bullish price target",
        f"{company_name} announces major partnership deal",
        f"{company_name} revenue growth exceeds Wall Street expectations",
        f"Investors cheer {company_name} buyback program expansion",
        f"{company_name} launches innovative new product line",
        f"{company_name} reports record-breaking sales figures",
    ]
    negative = [
        f"{company_name} misses earnings expectations, shares fall",
        f"Analysts downgrade {company_name} amid market concerns",
        f"{company_name} faces regulatory scrutiny over practices",
        f"{company_name} warns of slowing growth in key markets",
        f"Supply chain issues weigh on {company_name} outlook",
        f"{company_name} stock drops on disappointing guidance",
        f"Investors sell off {company_name} on macro headwinds",
    ]
    neutral = [
        f"{company_name} to report quarterly earnings next week",
        f"{company_name} CEO speaks at industry conference",
        f"{company_name} maintains market position in competitive sector",
        f"Market watches {company_name} amid sector rotation",
        f"{company_name} files standard regulatory documents",
    ]
    pool = positive * 3 + negative * 2 + neutral * 2
    np.random.shuffle(pool)
    selected  = pool[:n]
    base_date = datetime.now()
    dates     = [base_date - timedelta(days=i * 3) for i in range(len(selected))]
    return selected, dates


def fetch_headlines_rss(ticker, company_name, max_items=40):
    headlines = []
    dates     = []

    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            content = response.read().decode('utf-8')

        titles    = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', content)
        pub_dates = re.findall(r'<pubDate>(.*?)</pubDate>', content)

        for i, title in enumerate(titles[1:max_items+1]):
            headlines.append(title)
            if i < len(pub_dates) - 1:
                try:
                    dt = datetime.strptime(pub_dates[i+1][:16], '%a, %d %b %Y')
                    dates.append(dt)
                except:
                    dates.append(datetime.now() - timedelta(days=i))
            else:
                dates.append(datetime.now() - timedelta(days=i))

        print(f"  {len(headlines)} headlines fetched for {company_name} ({ticker})")

        if len(headlines) == 0:
            print(f"  No headlines found, using simulated headlines...")
            headlines, dates = simulate_headlines(company_name, max_items)

    except Exception as e:
        print(f"  RSS fetch failed ({e}), using simulated headlines...")
        headlines, dates = simulate_headlines(company_name, max_items)

    return headlines, dates

# ============================================================
# 3. VADER SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    results  = []

    for headline in headlines:
        scores   = analyzer.polarity_scores(headline)
        compound = scores['compound']

        if compound >= BUY_THRESHOLD:
            signal = 'BUY'
        elif compound <= SELL_THRESHOLD:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        results.append({
            'Headline': headline,
            'Compound': compound,
            'Positive': scores['pos'],
            'Negative': scores['neg'],
            'Neutral':  scores['neu'],
            'Signal':   signal
        })

    return pd.DataFrame(results)

# ============================================================
# 4. MERGE SENTIMENT WITH PRICE DATA
# ============================================================

def merge_sentiment_prices(sentiment_df, dates, prices, ticker):
    sent_df         = sentiment_df.copy()
    sent_df['Date'] = [d.date() for d in dates]

    daily_sent          = sent_df.groupby('Date')['Compound'].mean().reset_index()
    daily_sent.columns  = ['Date', 'AvgSentiment']
    daily_sent['Date']  = pd.to_datetime(daily_sent['Date'])

    price_df            = prices[[ticker]].copy().reset_index()
    price_df.columns    = ['Date', 'Close']
    price_df['NextDayReturn'] = price_df['Close'].pct_change().shift(-1) * 100

    merged = pd.merge(daily_sent, price_df, on='Date', how='inner')
    merged['Signal'] = merged['AvgSentiment'].apply(
        lambda x: 'BUY' if x >= BUY_THRESHOLD else ('SELL' if x <= SELL_THRESHOLD else 'HOLD')
    )

    return merged

# ============================================================
# 5. BACKTEST
# ============================================================

def backtest_signals(merged):
    results = []
    for _, row in merged.dropna(subset=['NextDayReturn']).iterrows():
        signal = row['Signal']
        ret    = row['NextDayReturn']
        if signal == 'BUY':
            correct = ret > 0
        elif signal == 'SELL':
            correct = ret < 0
        else:
            correct = None
        results.append(correct)

    results_clean = [r for r in results if r is not None]
    accuracy = np.mean(results_clean) * 100 if results_clean else 0
    return accuracy, results_clean

# ============================================================
# 6. PRINT SUMMARY
# ============================================================

def print_summary(sentiment_df, merged, accuracy):
    total = len(sentiment_df)
    buys  = (sentiment_df['Signal'] == 'BUY').sum()
    sells = (sentiment_df['Signal'] == 'SELL').sum()
    holds = (sentiment_df['Signal'] == 'HOLD').sum()
    avg   = sentiment_df['Compound'].mean()

    print("\n" + "="*55)
    print("   SENTIMENT TRADING SIGNAL — RESULTS")
    print("="*55)
    print(f"\n  Headlines Analyzed : {total}")
    print(f"  Avg Sentiment Score: {avg:+.3f}")
    print(f"\n  BUY  signals : {buys}  ({buys/total*100:.1f}%)")
    print(f"  HOLD signals : {holds} ({holds/total*100:.1f}%)")
    print(f"  SELL signals : {sells} ({sells/total*100:.1f}%)")
    print(f"\n  Signal Accuracy (backtest): {accuracy:.1f}%")
    print(f"\n  Top 5 Most Positive Headlines:")
    for _, row in sentiment_df.nlargest(5, 'Compound').iterrows():
        print(f"    [{row['Signal']:4s}] {row['Compound']:+.2f}  {row['Headline'][:60]}")
    print(f"\n  Top 5 Most Negative Headlines:")
    for _, row in sentiment_df.nsmallest(5, 'Compound').iterrows():
        print(f"    [{row['Signal']:4s}] {row['Compound']:+.2f}  {row['Headline'][:60]}")
    print("="*55)

# ============================================================
# 7. DASHBOARD
# ============================================================

def plot_dashboard(sentiment_df, merged, prices, accuracy, tickers):

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
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['text'], labelsize=8)
        ax.set_title(title, color=C['text'], fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=C['grid'], linewidth=0.5, linestyle='--')
        for sp in ax.spines.values():
            sp.set_color(C['grid'])

    # ── Panel 1: Price + Sentiment Overlay ──
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, f'{MAIN_NAME} ({MAIN_TICKER})  —  Price & Sentiment Signal')
    ticker_prices = prices[MAIN_TICKER]
    ax1.plot(ticker_prices.index, ticker_prices.values,
             color=C['blue'], lw=1.5, label='Price', zorder=3)
    ax1.set_ylabel('Price (USD)', color=C['text'], fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    if not merged.empty:
        ax1b = ax1.twinx()
        ax1b.set_facecolor(C['panel'])
        colors_sent = [C['green'] if s >= BUY_THRESHOLD else
                       (C['red'] if s <= SELL_THRESHOLD else C['yellow'])
                       for s in merged['AvgSentiment']]
        ax1b.bar(merged['Date'], merged['AvgSentiment'],
                 color=colors_sent, alpha=0.4, width=1.5)
        ax1b.axhline(BUY_THRESHOLD,  color=C['green'], lw=0.8, ls='--', alpha=0.6)
        ax1b.axhline(SELL_THRESHOLD, color=C['red'],   lw=0.8, ls='--', alpha=0.6)
        ax1b.set_ylabel('Sentiment Score', color=C['text'], fontsize=9)
        ax1b.tick_params(colors=C['text'], labelsize=8)
        ax1b.set_ylim(-1, 1)

    ax1.legend(loc='upper left', fontsize=8, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 2: Sentiment Distribution ──
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, 'Sentiment Score Distribution')
    for label, color in [('BUY', C['green']), ('HOLD', C['yellow']), ('SELL', C['red'])]:
        subset = sentiment_df[sentiment_df['Signal'] == label]['Compound']
        if len(subset):
            ax2.hist(subset, bins=15, color=color, alpha=0.7, label=label, density=True)
    ax2.axvline(BUY_THRESHOLD,  color=C['green'], lw=1, ls='--')
    ax2.axvline(SELL_THRESHOLD, color=C['red'],   lw=1, ls='--')
    ax2.set_xlabel('Compound Sentiment Score', color=C['text'], fontsize=9)
    ax2.set_ylabel('Density', color=C['text'], fontsize=9)
    ax2.legend(fontsize=8, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 3: Signal Pie ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C['panel'])
    ax3.set_title('Signal Breakdown', color=C['text'], fontsize=10, fontweight='bold', pad=8)
    signal_counts = sentiment_df['Signal'].value_counts()
    pie_colors    = {'BUY': C['green'], 'HOLD': C['yellow'], 'SELL': C['red']}
    colors_pie    = [pie_colors.get(s, C['blue']) for s in signal_counts.index]
    wedges, texts, autotexts = ax3.pie(
        signal_counts.values,
        labels=signal_counts.index,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'color': C['text'], 'fontsize': 10}
    )
    for at in autotexts:
        at.set_color(C['bg'])
        at.set_fontweight('bold')

    # ── Panel 4: Backtest Accuracy ──
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, 'Signal Backtest — Accuracy')

    buy_correct  = []
    sell_correct = []
    for _, row in merged.dropna(subset=['NextDayReturn']).iterrows():
        if row['Signal'] == 'BUY':
            buy_correct.append(row['NextDayReturn'] > 0)
        elif row['Signal'] == 'SELL':
            sell_correct.append(row['NextDayReturn'] < 0)

    buy_acc  = np.mean(buy_correct)  * 100 if buy_correct  else 0
    sell_acc = np.mean(sell_correct) * 100 if sell_correct else 0

    categories = ['BUY Accuracy', 'SELL Accuracy', 'Overall Accuracy']
    vals       = [buy_acc, sell_acc, accuracy]
    colors_b   = [C['green'] if v >= 50 else C['red'] for v in vals]
    bars       = ax4.bar(categories, vals, color=colors_b, alpha=0.8)
    ax4.axhline(50, color=C['yellow'], lw=1, ls='--', label='Random (50%)')
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', color=C['text'], fontsize=9, fontweight='bold')
    ax4.set_ylim(0, 110)
    ax4.set_ylabel('Accuracy (%)', color=C['text'], fontsize=9)
    ax4.legend(fontsize=8, facecolor=C['panel'], labelcolor=C['text'])

    # ── Panel 5: Multi-stock Sentiment Comparison ──
    ax5 = fig.add_subplot(gs[2, 1])
    style(ax5, 'Sentiment Comparison — All Stocks')
    analyzer     = SentimentIntensityAnalyzer()
    stock_scores = {}

    for name, ticker in tickers.items():
        headlines, _ = fetch_headlines_rss(ticker, name, max_items=20)
        scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        stock_scores[name] = np.mean(scores) if scores else 0

    names      = list(stock_scores.keys())
    values     = list(stock_scores.values())
    colors_bar = [C['green'] if v >= BUY_THRESHOLD else
                  (C['red'] if v <= SELL_THRESHOLD else C['yellow'])
                  for v in values]
    bars = ax5.bar(names, values, color=colors_bar, alpha=0.8)
    ax5.axhline(BUY_THRESHOLD,  color=C['green'], lw=1, ls='--', alpha=0.7, label='Buy zone')
    ax5.axhline(SELL_THRESHOLD, color=C['red'],   lw=1, ls='--', alpha=0.7, label='Sell zone')
    ax5.axhline(0, color=C['text'], lw=0.5, ls=':', alpha=0.4)
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.005 if val >= 0 else -0.015),
                 f'{val:+.3f}', ha='center', color=C['text'], fontsize=8)
    ax5.set_ylabel('Avg Sentiment Score', color=C['text'], fontsize=9)
    ax5.set_ylim(-0.5, 0.5)
    ax5.legend(fontsize=7, facecolor=C['panel'], labelcolor=C['text'])
    plt.setp(ax5.get_xticklabels(), rotation=15, ha='right')

    fig.suptitle('SENTIMENT TRADING SIGNAL  —  NLP + MARKET ANALYSIS',
                 color=C['text'], fontsize=16, fontweight='bold', y=0.985)

    plt.savefig('sentiment_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print("\nDashboard saved as sentiment_dashboard.png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*55)
    print("  SENTIMENT TRADING SIGNAL")
    print("  Finance Portfolio Project #6")
    print("="*55 + "\n")

    prices = fetch_prices(TICKERS, PERIOD)

    print(f"Fetching headlines for {MAIN_NAME} ({MAIN_TICKER})...")
    headlines, dates = fetch_headlines_rss(MAIN_TICKER, MAIN_NAME, max_items=40)

    print("\nRunning VADER sentiment analysis...")
    sentiment_df = analyze_sentiment(headlines)
    print(f"  Done. {len(sentiment_df)} headlines scored.\n")

    merged = merge_sentiment_prices(sentiment_df, dates, prices, MAIN_TICKER)
    accuracy, _ = backtest_signals(merged)

    print_summary(sentiment_df, merged, accuracy)

    print("\nGenerating dashboard...")
    plot_dashboard(sentiment_df, merged, prices, accuracy, TICKERS)

    print("\n" + "="*55)
    print("  ALL 6 PROJECTS COMPLETE!")
    print("="*55)