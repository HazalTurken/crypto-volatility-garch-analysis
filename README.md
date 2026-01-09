# Crypto Volatility & Risk Analysis  
### Bitcoin vs Ethereum

This project provides a comparative risk and volatility analysis of **Bitcoin (BTC)** and **Ethereum (ETH)** using financial time-series techniques commonly applied in asset management and quantitative finance.


## Objectives
- Analyze and compare volatility dynamics of BTC and ETH
- Examine return distributions and tail risk
- Model volatility persistence using **GARCH(1,1)**
- Quantify downside risk using **Value at Risk (VaR)**


## Methods Used
- Daily log returns
- 30-day rolling volatility
- Return distribution analysis (skewness & kurtosis)
- Value at Risk (95%)
- GARCH(1,1) with Student’s t distribution


## Key Findings
- Ethereum exhibits **consistently higher volatility** than Bitcoin
- Both assets show **volatility clustering and persistence**
- Return distributions are **non-normal with heavy tails**
- Ethereum carries **higher downside risk (VaR)**
- Moderate correlation (~0.5) suggests partial diversification benefits
  

## Practical Implications
- Bitcoin behaves as a relatively more stable crypto asset
- Ethereum offers higher return potential but significantly higher risk
- Risk models assuming normality underestimate crypto tail risk
- GARCH models are effective for crypto volatility forecasting
  

## Tools & Libraries
- Python
- pandas, numpy
- matplotlib, seaborn
- scipy
- arch (GARCH modeling)


## Project Structure
├── data/
├── notebooks/
│ └── crypto_volatility_analysis.ipynb
├── outputs/
│ └── figures/
├── README.md
└── requirements.txt


## Author
Hazal Turken  
Master of Business Analytics & Artificial Intelligence (MBAI)
