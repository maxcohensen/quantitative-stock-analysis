import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ruptures import Pelt
from ruptures.costs import CostL1
from scipy.stats import norm
from datetime import date

# Generating synthetic data for illustration
# np.random.seed(0) IMPLEMENT FOR TESTING
dates = pd.date_range(start='2023-01-01', end=date.today(), freq='D')
n = len(dates)
close_prices = np.random.uniform(100, 200, n)

data = pd.DataFrame({
    'Date': dates,
    'Close': close_prices
})

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate Bollinger Bands
window = 20
data['MA'] = data['Close'].rolling(window=window).mean()
data['STD'] = data['Close'].rolling(window=window).std()
data['Upper_Band'] = data['MA'] + (data['STD'] * 2)
data['Lower_Band'] = data['MA'] - (data['STD'] * 2)
data.dropna(inplace=True)

# Changepoint Detection
algo = Pelt(model="l1").fit(data['Close'].values)
result = algo.predict(pen=1)
changepoints = [data.index[i] for i in result if i < len(data.index)]

# Random Forest for prediction
features = ['MA', 'Upper_Band', 'Lower_Band']
X = data[features].values
y = data['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Monte Carlo simulation
S0 = data['Close'].iloc[-1]
T = 8
mu = np.mean(data['Close'].pct_change())
sigma = np.std(data['Close'].pct_change())
iterations = 10000

paths = np.zeros((T, iterations))
paths[0] = S0  # Initializing the first row to S0

dt = 1/252  # Assuming 252 trading days in a year
for t in range(1, T):
    rand_values = np.random.normal(size=iterations)
    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand_values)

'''
# Trading Logic
capital = 10000  # Starting capital USD
print(f"Starting capital: {capital}")

position = 0  # No position initially

for index, row in data.iterrows():
    if row['Close'] < row['Lower_Band']:
        position = capital // row['Close']
        capital -= position * row['Close']
    elif row['Close'] > row['Upper_Band']:
        capital += position * row['Close']
        position = 0

final_capital = capital + position * data['Close'].iloc[-1]
print(f"Final capital: {final_capital}")
'''

# Visualization
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.title('Stock Price and Bollinger Bands')
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Upper_Band'], label='Upper Bollinger Band')
plt.plot(data['Lower_Band'], label='Lower Bollinger Band')
plt.scatter(changepoints, [data.loc[p]['Close'] for p in changepoints], c='r', label='Changepoints')

# For Monte Carlo Simulation
plt.subplot(3, 1, 2)
plt.title('Monte Carlo Simulation')
plt.xlabel('Future Trading Days')
plt.ylabel('Simulated Stock Price')
plt.plot(paths)
plt.xlim([0, T])  # T is the time-horizon

# For Histogram
plt.subplot(3, 1, 3)
plt.title('Histogram of Stock Price at End of Simulations')
plt.xlabel('Stock Price')
plt.ylabel('Density')
plt.hist(paths[-1], bins=50, density=True)

plt.tight_layout()
plt.show()
