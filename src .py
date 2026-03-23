import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_var():
    tickers = ['AAPL', 'MSFT', 'GOOG']
    data = yf.download(tickers, start='2023-01-01', end='2024-12-31')['Close']
    
    returns = data.pct_change().dropna()
    
    weights = np.array([0.4, 0.3, 0.3])
    portfolio_returns = returns @ weights
    
    confidence_level = 0.95
    
    # Historical VaR
    var_hist = np.percentile(portfolio_returns, 5)
    
    # Parametric VaR
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    z = norm.ppf(0.05)
    var_param = mean + z * std
    
    return var_hist, var_param

if __name__ == "__main__":
    var_hist, var_param = calculate_var()
    print("Historical VaR:", var_hist)
    print("Parametric VaR:", var_param)
