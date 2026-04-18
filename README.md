# 📈 Python Finance Portfolio

A collection of 6 finance + machine learning projects built in Python, covering quantitative analysis, risk modeling, machine learning, and NLP-based trading signals.

---

## 🗂️ Projects

### 1. Stock Portfolio Analyzer
**File:** `stock_analyzer.py`

Analyzes a multi-stock portfolio using real market data fetched via `yfinance`.

**Features:**
- Sharpe Ratio, Maximum Drawdown, Value at Risk (VaR)
- 20/50-day Moving Averages with Buy/Sell signals
- 5-panel dark-themed dashboard (price, returns, drawdown, correlation, performance)

**Libraries:** `yfinance`, `pandas`, `numpy`, `matplotlib`

---

### 2. ML Stock Price Predictor
**File:** `stock_predictor.py`

Predicts future stock prices using supervised machine learning models trained on historical AAPL data.

**Features:**
- Linear Regression + Random Forest Regressor
- Feature engineering: lag returns, moving averages, volatility
- R² score of 0.73 on test data
- Actual vs predicted price chart

**Libraries:** `scikit-learn`, `yfinance`, `pandas`, `numpy`, `matplotlib`

---

### 3. Crypto Market Analyzer
**File:** `crypto_analyzer.py`

Real-time analysis of 5 major cryptocurrencies: BTC, ETH, SOL, BNB, ADA.

**Features:**
- RSI (14-day) with overbought/oversold zones
- Bollinger Bands (20-day)
- Volume analysis with 20-day average
- Correlation heatmap across all 5 coins
- Normalized performance comparison chart

**Libraries:** `yfinance`, `pandas`, `numpy`, `matplotlib`

---

### 4. Credit Risk Model
**File:** `credit_risk_model.py`

Predicts loan default risk using synthetic applicant data modeled on 10 major US banks.

**Features:**
- 2,000 synthetic loan applicants across Bank of America, JPMorgan Chase, Wells Fargo, Goldman Sachs, and 6 more
- FICO credit scores, debt-to-income ratio, employment history
- Logistic Regression vs Random Forest classifier
- ROC Curve + AUC, Confusion Matrix, Feature Importance
- Default rate breakdown by bank

**Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---

### 5. Monte Carlo Simulation
**File:** `monte_carlo_simulation.py`

Simulates 1,000 portfolio scenarios over a 1-year horizon using correlated asset returns.

**Features:**
- Cholesky decomposition to preserve asset correlations
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional VaR / Expected Shortfall (CVaR)
- Efficient Frontier with Sharpe Ratio heatmap
- Percentile bands: 5th, 25th, median, 75th, 95th

**Portfolio:** AAPL, MSFT, GOOGL, AMZN, NVDA | Initial: $100,000

**Libraries:** `yfinance`, `pandas`, `numpy`, `matplotlib`

---

### 6. Sentiment Trading Signal
**File:** `sentiment_trading_signal.py`

NLP-powered trading signal system that maps financial news sentiment to BUY / HOLD / SELL signals.

**Features:**
- VADER sentiment analysis on live financial headlines
- Yahoo Finance RSS feed integration with simulated fallback
- Signal backtesting: does sentiment predict next-day price movement?
- Sentiment comparison across AAPL, MSFT, NVDA, AMZN, TSLA
- BUY threshold: +0.15 | SELL threshold: -0.15

**Libraries:** `vaderSentiment`, `yfinance`, `pandas`, `numpy`, `matplotlib`

---

## 🛠️ Setup

```bash
git clone https://github.com/Attack-1123/stock-portfolio-analyzer.git
cd stock-portfolio-analyzer
pip install yfinance pandas numpy matplotlib scikit-learn vaderSentiment
```

Run any project:
```bash
python stock_analyzer.py
python stock_predictor.py
python crypto_analyzer.py
python credit_risk_model.py
python monte_carlo_simulation.py
python sentiment_trading_signal.py
```

---

## 📊 Tech Stack

| Category | Libraries |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| Machine Learning | `scikit-learn` |
| NLP | `vaderSentiment` |
| Visualization | `matplotlib` |
| Language | Python 3.14 |

---

## 👤 Author

**Mustafa** — Finance & AI/ML enthusiast building toward a career in quantitative finance.

GitHub: [@Attack-1123](https://github.com/Attack-1123)