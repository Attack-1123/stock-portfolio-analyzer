import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ticker        = "AAPL"
period        = "6mo"
risk_free     = 0.05

print(f"\nFetching data for {ticker}...")
stock  = yf.download(ticker, period=period, auto_adjust=True)
closes = stock["Close"].dropna().values.flatten()
dates  = stock.index[:len(closes)]
print(f"Got {len(closes)} trading days of data!\n")

daily_returns   = np.diff(closes) / closes[:-1]
mean_daily      = np.mean(daily_returns)
std_daily       = np.std(daily_returns)

annual_return   = mean_daily * 252 * 100
annual_std      = std_daily  * np.sqrt(252) * 100
sharpe          = (annual_return/100 - risk_free) / (annual_std/100)

peak            = np.maximum.accumulate(closes)
drawdown        = (closes - peak) / peak * 100
max_drawdown    = np.min(drawdown)
var_95          = np.percentile(daily_returns, 5) * 100

def moving_avg(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

ma20  = moving_avg(closes, 20)
ma50  = moving_avg(closes, 50)

offset   = len(closes) - len(ma50)
ma_dates = dates[offset:]
ma20_al  = ma20[len(ma20)-len(ma50):]

signals  = np.diff(np.sign(ma20_al - ma50))
buy_idx  = np.where(signals > 0)[0] + 1
sell_idx = np.where(signals < 0)[0] + 1

compare_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
comp_returns    = {}

print("Fetching comparison data...")
for t in compare_tickers:
    d = yf.download(t, period=period, auto_adjust=True, progress=False)
    c = d["Close"].dropna().values.flatten()
    r = np.diff(c) / c[:-1]
    comp_returns[t] = {
        "annual_return" : np.mean(r) * 252 * 100,
        "annual_std"    : np.std(r)  * np.sqrt(252) * 100,
        "sharpe"        : (np.mean(r)*252 - risk_free) / (np.std(r)*np.sqrt(252)),
        "total_return"  : (c[-1] - c[0]) / c[0] * 100
    }

print("\n" + "=" * 42)
print(f"   {ticker} — Full Risk Analysis ({period})")
print("=" * 42)
print(f"  Current Price    : ${closes[-1]:.2f}")
print(f"  Annual Return    : {annual_return:+.2f}%")
print(f"  Annual Volatility: {annual_std:.2f}%")
print(f"  Sharpe Ratio     : {sharpe:.3f}  {'✓ Good' if sharpe > 1 else '✗ Weak'}")
print(f"  Max Drawdown     : {max_drawdown:.2f}%")
print(f"  Value at Risk 95%: {var_95:.2f}% per day")
print("=" * 42)

print("\n── Multi-Stock Comparison ──────────────────")
print(f"  {'Ticker':<8}{'Return':>10}{'Volatility':>12}{'Sharpe':>10}")
print("  " + "-" * 40)
for t, m in comp_returns.items():
    print(f"  {t:<8}{m['annual_return']:>9.1f}%{m['annual_std']:>11.1f}%{m['sharpe']:>10.3f}")
print("=" * 42)
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"{ticker} — Portfolio Analysis Dashboard", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(dates, closes, color="steelblue", linewidth=1.2, label="Price", zorder=1)
ax1.plot(ma_dates, ma50, color="orange", linewidth=1.5, label="MA50")
ax1.plot(ma_dates, ma20_al, color="green", linewidth=1.2, label="MA20", linestyle="--")
ax1.scatter(ma_dates[buy_idx], closes[offset+buy_idx], marker="^", color="lime", s=80, zorder=5, label="Buy signal")
ax1.scatter(ma_dates[sell_idx], closes[offset+sell_idx], marker="v", color="red", s=80, zorder=5, label="Sell signal")
ax1.set_title("Price + Moving Averages + Buy/Sell Signals")
ax1.set_ylabel("Price ($)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
colors = ["green" if r >= 0 else "red" for r in daily_returns]
ax2.bar(dates[1:], daily_returns*100, color=colors, width=1)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Daily Returns (%)")
ax2.set_ylabel("Return (%)")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.fill_between(dates, drawdown, 0, color="red", alpha=0.4)
ax3.set_title(f"Drawdown (Max: {max_drawdown:.1f}%)")
ax3.set_ylabel("Drawdown (%)")
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
tickers_list = list(comp_returns.keys())
returns_list = [comp_returns[t]["annual_return"] for t in tickers_list]
bar_colors   = ["steelblue" if t != ticker else "orange" for t in tickers_list]
ax4.bar(tickers_list, returns_list, color=bar_colors)
ax4.axhline(0, color="black", linewidth=0.8)
ax4.set_title("Annual Return Comparison")
ax4.set_ylabel("Return (%)")
ax4.grid(True, alpha=0.3, axis="y")

ax5 = fig.add_subplot(gs[2, 1])
sharpe_list = [comp_returns[t]["sharpe"] for t in tickers_list]
sharpe_colors = ["green" if s > 1 else "salmon" for s in sharpe_list]
ax5.bar(tickers_list, sharpe_list, color=sharpe_colors)
ax5.axhline(1, color="orange", linewidth=1.2, linestyle="--", label="Sharpe = 1.0")
ax5.set_title("Sharpe Ratio Comparison")
ax5.set_ylabel("Sharpe Ratio")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

plt.savefig("portfolio_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as portfolio_analysis.png")