import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Loading Bitcoin Data
btc_df = pd.read_csv("C:/Users/hazal/OneDrive/Masaüstü/bitcoin.csv")
btc_df['Date'] = pd.to_datetime(btc_df['Timestamp'], unit='s')
btc_df = btc_df.sort_values('Date')[['Date', 'Close']]

# Loading Ethereum data
eth_df = pd.read_csv("C:/Users/hazal/OneDrive/Masaüstü/ETH.csv", sep=None, engine="python")

# Find time column for Ethereum
time_candidates = {
    "timestamp","time","date","datetime","open time","close time","open_time","close_time","unix"
}
time_col = None
for c in eth_df.columns:
    if c.strip().lower() in time_candidates:
        time_col = c
        break
if time_col is None:
    time_col = eth_df.columns[0]

# Convert to datetime for Ethereum
s = eth_df[time_col]
if np.issubdtype(s.dtype, np.number):
    unit = "ms" if s.dropna().astype(float).median() > 1e12 else "s"
    dt = pd.to_datetime(s, unit=unit, errors="coerce")
else:
    dt = pd.to_datetime(s, errors="coerce")
eth_df["Date"] = dt

# Find Close column for Ethereum
close_candidates = {
    "close","close*","closing price","close price","price","last","adj close","adj_close"
}
close_col = None
for c in eth_df.columns:
    if c.strip().lower() in close_candidates:
        close_col = c
        break
if close_col is None:
    for c in eth_df.columns:
        if "close" in c.strip().lower():
            close_col = c
            break
if close_col is None:
    raise ValueError("Close column not found for Ethereum data.")

# Clean Ethereum data
eth_df = eth_df[["Date", close_col]].dropna()
eth_df.rename(columns={close_col:"Close"}, inplace=True)
eth_df["Close"] = pd.to_numeric(eth_df["Close"], errors="coerce")
eth_df = eth_df.dropna(subset=["Date","Close"]).sort_values("Date")

# Daily Resampling and Returns
btc_day = btc_df.set_index("Date").resample("1D").last().dropna()
eth_day = eth_df.set_index("Date").resample("1D").last().dropna()

btc_day["Return"] = np.log(btc_day["Close"] / btc_day["Close"].shift(1))
eth_day["Return"] = np.log(eth_day["Close"] / eth_day["Close"].shift(1))

# Calculate 30-Day Volatility
btc_day["Volatility_30d"] = btc_day["Return"].rolling(30).std()
eth_day["Volatility_30d"] = eth_day["Return"].rolling(30).std()

# Find Common Date Range for Comparison
common_start = max(btc_day.index.min(), eth_day.index.min())
common_end = min(btc_day.index.max(), eth_day.index.max())

btc_common = btc_day[(btc_day.index >= common_start) & (btc_day.index <= common_end)]
eth_common = eth_day[(eth_day.index >= common_start) & (eth_day.index <= common_end)]

common_dates = btc_common.index.intersection(eth_common.index)
btc_aligned = btc_common.loc[common_dates]
eth_aligned = eth_common.loc[common_dates]

# Plot Comparison

# 1. Individual Volatility Plots
plt.figure(figsize=(15, 10))

# Bitcoin Volatility
plt.subplot(2, 1, 1)
plt.plot(btc_day.index, btc_day["Volatility_30d"], label="Bitcoin 30-Day Volatility", linewidth=2, alpha=0.8, color='orange')
plt.title("Bitcoin Volatility (30-Day Rolling Std)", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Ethereum Volatility
plt.subplot(2, 1, 2)
plt.plot(eth_day.index, eth_day["Volatility_30d"], label="Ethereum 30-Day Volatility", linewidth=2, alpha=0.8, color='blue')
plt.title("Ethereum Volatility (30-Day Rolling Std)", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Combined Volatility Comparison
plt.figure(figsize=(15, 8))
plt.plot(btc_aligned.index, btc_aligned["Volatility_30d"], label="Bitcoin 30-Day Volatility", linewidth=2, alpha=0.8, color='orange')
plt.plot(eth_aligned.index, eth_aligned["Volatility_30d"], label="Ethereum 30-Day Volatility", linewidth=2, alpha=0.8, color='blue')
plt.title("Bitcoin vs Ethereum Volatility Comparison (30-Day Rolling Std)", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Return Distribution Comparison
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
btc_aligned["Return"].dropna().hist(bins=100, alpha=0.7, color='orange', label='Bitcoin')
plt.title("Bitcoin Daily Return Distribution", fontsize=14)
plt.xlabel("Daily Return", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
eth_aligned["Return"].dropna().hist(bins=100, alpha=0.7, color='blue', label='Ethereum')
plt.title("Ethereum Daily Return Distribution", fontsize=14)
plt.xlabel("Daily Return", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Overlapping Return Distributions
plt.figure(figsize=(12, 6))
btc_aligned["Return"].dropna().hist(bins=100, alpha=0.6, color='orange', label='Bitcoin', density=True)
eth_aligned["Return"].dropna().hist(bins=100, alpha=0.6, color='blue', label='Ethereum', density=True)
plt.title("Bitcoin vs Ethereum Daily Return Distribution Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Daily Return", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistical Comparison
print("\n" + "="*60)
print("STATISTICAL COMPARISON")
print("="*60)

btc_returns_full = btc_day["Return"].dropna()
eth_returns_full = eth_day["Return"].dropna()
btc_returns = btc_aligned["Return"].dropna()
eth_returns = eth_aligned["Return"].dropna()

print(f"\nBitcoin Statistics:")
print(f"Mean Daily Return: {btc_returns.mean():.4f} ({btc_returns.mean()*100:.2f}%)")
print(f"Daily Volatility: {btc_returns.std():.4f} ({btc_returns.std()*100:.2f}%)")
print(f"Skewness: {btc_returns.skew():.4f}")
print(f"Kurtosis: {btc_returns.kurtosis():.4f}")

print(f"\nEthereum Statistics:")
print(f"Mean Daily Return: {eth_returns.mean():.4f} ({eth_returns.mean()*100:.2f}%)")
print(f"Daily Volatility: {eth_returns.std():.4f} ({eth_returns.std()*100:.2f}%)")
print(f"Skewness: {eth_returns.skew():.4f}")
print(f"Kurtosis: {eth_returns.kurtosis():.4f}")

# VaR Comparison
alpha = 0.05
btc_var = np.percentile(btc_returns, 100*alpha)
eth_var = np.percentile(eth_returns, 100*alpha)

print(f"\nValue at Risk (95%):")
print(f"Bitcoin 95% VaR: {btc_var:.4f} ({btc_var*100:.2f}%)")
print(f"Ethereum 95% VaR: {eth_var:.4f} ({eth_var*100:.2f}%)")

# Correlation
correlation = btc_aligned["Return"].corr(eth_aligned["Return"])
print(f"\nCorrelation between Bitcoin and Ethereum returns: {correlation:.4f}")

# GARCH Model Comparison
print("\n" + "="*60)
print("GARCH MODEL COMPARISON")
print("="*60)

# Bitcoin GARCH
print("\nBitcoin GARCH(1,1) Model:")
btc_y = btc_returns_full.astype("float64") * 1000
btc_model = arch_model(btc_y, vol="Garch", p=1, q=1, mean="Constant", dist="t")
btc_res = btc_model.fit(disp="off")
print(btc_res.summary())

# Ethereum GARCH
print("\nEthereum GARCH(1,1) Model:")
eth_y = eth_returns_full.astype("float64") * 1000
eth_model = arch_model(eth_y, vol="Garch", p=1, q=1, mean="Constant", dist="t")
eth_res = eth_model.fit(disp="off")
print(eth_res.summary())

# GARCH Volatility Comparison
btc_garch_vol = btc_res.conditional_volatility / 1000
eth_garch_vol = eth_res.conditional_volatility / 1000

plt.figure(figsize=(15, 8))
plt.plot(btc_day.index[-len(btc_garch_vol):], btc_garch_vol, label="Bitcoin GARCH(1,1) Volatility", linewidth=2, alpha=0.8, color='orange')
plt.plot(eth_day.index[-len(eth_garch_vol):], eth_garch_vol, label="Ethereum GARCH(1,1) Volatility", linewidth=2, alpha=0.8, color='blue')
plt.title("Bitcoin vs Ethereum GARCH(1,1) Conditional Volatility Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Conditional Volatility", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nAnalysis completed!")
