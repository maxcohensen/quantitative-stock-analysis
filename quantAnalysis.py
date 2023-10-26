# Stock Price Analysis and Prediction

# Import Libraries
# Essential libraries for data manipulation and scientific computation
import numpy as np
import pandas as pd
# Libraries for machine learning and statistical models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
# Libraries for data visualization
import matplotlib.pyplot as plt
# Libraries for changepoint detection
from ruptures import Pelt
from ruptures.costs import CostL1
# Library for manipulating date objects
from datetime import date

# Generate Synthetic Data
# Create a synthetic dataset representing stock 'Close' prices with random values.
dates = pd.date_range(start='2023-01-01', end=date.today(), freq='D')
data = pd.DataFrame({
    'Date': dates,
    'Close': np.random.uniform(100, 200, len(dates))
})
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Bollinger Bands Calculation
# Rolling window size for Bollinger Bands
window = 20
data['MA'] = data['Close'].rolling(window=window).mean()
data['STD'] = data['Close'].rolling(window=window).std()
data['Upper_Band'] = data['MA'] + (data['STD'] * 2)
data['Lower_Band'] = data['MA'] - (data['STD'] * 2)
data.dropna(inplace=True)

# Changepoint Detection
# Implement Pelt algorithm for changepoint detection with L1 cost model
algo = Pelt(model="l1").fit(data['Close'].values)
result = algo.predict(pen=1)
changepoints = [data.index[i] for i in result if i < len(data.index)]

# Random Forest Regression for Stock Price Prediction
# Feature selection
features = ['MA', 'Upper_Band', 'Lower_Band']
X = data[features].values
y = data['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Monte Carlo Simulation for Future Stock Prices
# Perform a Monte Carlo simulation to forecast future stock prices
S0 = data['Close'].iloc[-1]
T = 8
mu = np.mean(data['Close'].pct_change())
sigma = np.std(data['Close'].pct_change())
iterations = 10000
paths = np.zeros((T, iterations))
paths[0] = S0
dt = 1/252
for t in range(1, T):
    rand_values = np.random.normal(size=iterations)
    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_values)

# Data Visualization
plt.figure(figsize=(14, 7))
plt.subplot(3, 1, 1)
plt.title('Stock Price and Bollinger Bands')
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Upper_Band'], label='Upper Bollinger Band')
plt.plot(data['Lower_Band'], label='Lower Bollinger Band')
plt.scatter(changepoints, [data.loc[p]['Close'] for p in changepoints], c='r', label='Changepoints')
plt.subplot(3, 1, 2)
plt.title('Monte Carlo Simulation')
plt.xlabel('Future Trading Days')
plt.ylabel('Simulated Stock Price')
plt.plot(paths)
plt.subplot(3, 1, 3)
plt.title('Histogram of Stock Price at End of Simulations')
plt.xlabel('Stock Price')
plt.ylabel('Density')
plt.hist(paths[-1], bins=50, density=True)
plt.tight_layout()
plt.show()
