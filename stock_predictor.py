import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── SETTINGS ─────────────────────────────────────────────────
ticker = "AAPL"   # change this to any stock
period = "1y"     # 1 year of data

# ── 1. FETCH DATA ─────────────────────────────────────────────
# yfinance goes to Yahoo Finance and downloads real stock data
print(f"\nFetching {ticker} data...")
stock = yf.download(ticker, period=period, auto_adjust=True)
df    = stock[["Close", "Volume"]].copy()  # we only need price and volume
df.columns = ["close", "volume"]           # rename to lowercase
df    = df.dropna()                        # remove any empty rows
print(f"Got {len(df)} days of data!\n")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
# Features are the "clues" we give the model to learn from
# The more meaningful clues, the smarter the model

df["returns"]    = df["close"].pct_change()          # % price change today
df["ma5"]        = df["close"].rolling(5).mean()     # average price last 5 days
df["ma20"]       = df["close"].rolling(20).mean()    # average price last 20 days
df["ma50"]       = df["close"].rolling(50).mean()    # average price last 50 days
df["volatility"] = df["returns"].rolling(20).std()   # how jumpy price has been
df["momentum"]   = df["close"] - df["close"].shift(5) # higher or lower than 5 days ago
df["target"]     = df["close"].shift(-1)             # tomorrows price (what we predict)
df = df.dropna()                                     # remove rows with empty values

# X = the clues (inputs), y = the answer (output we want to predict)
features = ["close", "volume", "returns", "ma5", "ma20", "ma50", "volatility", "momentum"]
X = df[features].values
y = df["target"].values

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────
# Golden rule of ML — never test on data you trained on
# 80% train, 20% test, shuffle=False keeps time order intact
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# StandardScaler converts all features to the same scale
# so the model treats each clue fairly
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn scale from training data
X_test  = scaler.transform(X_test)       # apply same scale to test data

print(f"Training on {len(X_train)} days")
print(f"Testing on  {len(X_test)} days\n")

# ── 4. TRAIN THE MODELS ───────────────────────────────────────
# fit() is the magic word — it means "learn from this data"

# Model 1: Linear Regression — finds the best straight line through data
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Model 2: Random Forest — builds 100 decision trees and averages them
# n_estimators=100 means 100 trees, random_state=42 makes results reproducible
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Both models trained!\n")

# ── 5. MAKE PREDICTIONS ───────────────────────────────────────
# Now models see data they have NEVER seen before and try to predict prices
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# ── 6. MEASURE ACCURACY ───────────────────────────────────────
# RMSE = average dollar error (lower = better)
# R2   = how well model explains price movements (1.0 = perfect, 0.0 = random)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
lr_r2   = r2_score(y_test, lr_pred)
rf_r2   = r2_score(y_test, rf_pred)

print("=" * 42)
print(f"   {ticker} — ML Prediction Results")
print("=" * 42)
print(f"  Linear Regression:")
print(f"    RMSE : ${lr_rmse:.2f} avg error per day")
print(f"    R²   : {lr_r2:.4f}")
print(f"  Random Forest:")
print(f"    RMSE : ${rf_rmse:.2f} avg error per day")
print(f"    R²   : {rf_r2:.4f}")
print("=" * 42)

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────
# Random Forest tells us which clues mattered most
# Higher value = more important for predictions
importance = rf_model.feature_importances_
feat_names  = features

print("\n── Feature Importance ──────────────────────")
for name, imp in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:<12} {bar} {imp:.4f}")

# ── 8. CHARTS ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle(f"{ticker} — ML Stock Price Predictor", fontsize=14, fontweight="bold")

test_dates = df.index[-len(y_test):]

# Chart 1 — Actual vs Linear Regression predictions
axes[0].plot(test_dates, y_test,   color="steelblue", linewidth=1.5, label="Actual price")
axes[0].plot(test_dates, lr_pred,  color="orange",    linewidth=1.2, linestyle="--", label=f"Linear Regression (R²={lr_r2:.3f})")
axes[0].set_title("Linear Regression — Predicted vs Actual")
axes[0].set_ylabel("Price ($)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Chart 2 — Actual vs Random Forest predictions
axes[1].plot(test_dates, y_test,   color="steelblue", linewidth=1.5, label="Actual price")
axes[1].plot(test_dates, rf_pred,  color="green",     linewidth=1.2, linestyle="--", label=f"Random Forest (R²={rf_r2:.3f})")
axes[1].set_title("Random Forest — Predicted vs Actual")
axes[1].set_ylabel("Price ($)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Chart 3 — Feature importance bar chart
colors = ["steelblue" if i == np.argmax(importance) else "lightsteelblue" for i in range(len(feat_names))]
axes[2].barh(feat_names, importance, color=colors)
axes[2].set_title("Feature Importance — Which clues mattered most?")
axes[2].set_xlabel("Importance Score")
axes[2].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("ml_predictor.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved as ml_predictor.png")