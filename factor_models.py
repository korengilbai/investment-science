from cmath import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import numpy as np
from sklearn.linear_model import LinearRegression
from functools import reduce
import statsmodels.api as sm

tickers = ["MSFT", "BRK-B", "GOOG", "NVO", "AAPL"]
multpl_stocks = web.get_data_yahoo(tickers,
start = "2012-10-10",
end = "2022-10-17")

rf = web.get_data_yahoo("^IRX",
start = "2012-10-10",
end = "2022-10-16")

rf = rf['Adj Close'] / (100 * 252)

sp500 = web.get_data_yahoo("^GSPC",
start = "2012-10-10",
end = "2022-10-17")

xci = web.get_data_yahoo("^XCI",
start = "2012-10-10",
end = "2022-10-17")

vix = web.get_data_yahoo("^VIX",
start = "2012-10-10",
end = "2022-10-17")

momentum = pd.read_csv("Developed_ex_US_MOM_Factor_Daily.csv", parse_dates=['Date'])
momentum = momentum.iloc[5724:8338]

multpl_stock_daily_returns = multpl_stocks['Adj Close'].pct_change()[1:]
portfolio_daily_returns = multpl_stock_daily_returns.mean(axis=1)
portfolio_excess_daily_returns = portfolio_daily_returns - rf
sp_excess_daily_returns = sp500['Adj Close'].pct_change()[1:] - rf
xci_excess_daily_returns = xci['Adj Close'].pct_change()[1:] - rf
vix_excess_daily_returns = vix['Adj Close'].pct_change()[1:] - rf

sp_excess_daily_returns.rename('S&P', inplace=True)
xci_excess_daily_returns.rename('XCI', inplace=True)
vix_excess_daily_returns.rename('VIX', inplace=True)

data_frames = [sp_excess_daily_returns, xci_excess_daily_returns, vix_excess_daily_returns, momentum]
x = reduce(lambda left,right: pd.merge(left,right,on=['Date']), data_frames)[1:]
x = x.drop(columns=["Date"])
x = x.dropna(axis = 0)
y = portfolio_excess_daily_returns[1:]

x.dropna()
x = x.to_numpy()
y = y.to_numpy()
y = y[~np.isnan(y)]
y = y.reshape(y.shape[0], 1)

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")