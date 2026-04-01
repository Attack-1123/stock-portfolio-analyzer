# 📈 Stock Portfolio Analyzer

A Python project that fetches real stock market data and performs
professional financial analysis using NumPy, Pandas, and Matplotlib.

Built as my first Python finance project as a Finance student.

---

## 🔍 What It Does

- Fetches real-time stock data via Yahoo Finance (yfinance)
- Calculates key financial metrics using NumPy
- Compares multiple stocks side by side
- Generates a full 5-panel analysis dashboard

---

## 📊 Financial Metrics Calculated

| Metric | Description |
|---|---|
| Annual Return | Annualized % gain or loss |
| Sharpe Ratio | Return vs risk (above 1.0 = good) |
| Max Drawdown | Biggest peak-to-trough drop |
| Volatility | Standard deviation of returns |
| Value at Risk | Worst expected daily loss (95% confidence) |

---

## 📉 Charts Generated

1. Price chart with 20-day and 50-day moving averages
2. Buy / Sell signals based on MA crossovers
3. Daily returns (green = positive, red = negative)
4. Drawdown chart
5. Multi-stock return and Sharpe ratio comparison

---

## 🛠️ Libraries Used

- `numpy` — financial calculations
- `yfinance` — fetches live stock data
- `pandas` — data manipulation
- `matplotlib` — charting and visualization

---

## ▶️ How To Run

**1. Install dependencies:**
```
pip install numpy yfinance pandas matplotlib
```

**2. Run the analyzer:**
```
python stock_analyzer.py
```

**3. Change the stock:**

Edit line 8 in `stock_analyzer.py`:
```python
ticker = "AAPL"  # Change to any stock e.g. GOOGL, TSLA, MSFT
```

---

## 📸 Sample Output
```
==========================================
   AAPL — Full Risk Analysis (6mo)
==========================================
  Current Price    : $253.79
  Annual Return    : +1.36%
  Annual Volatility: 21.40%
  Sharpe Ratio     : -0.170  ✗ Weak
  Max Drawdown     : -13.80%
  Value at Risk 95%: -1.92% per day
==========================================
```

---

## 👤 Author

Muhammad Mustafa — Finance Student | Python intergration