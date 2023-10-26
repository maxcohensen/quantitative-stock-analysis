# Stock-Price-Modeling-Suite

## Overview

The Stock-Price-Modeling-Suite is a comprehensive toolkit designed for stock market analysis and prediction. It integrates machine learning models, Monte Carlo simulations, and change-point detection algorithms to offer a multi-faceted approach to financial analytics.

## Features

- Bollinger Bands Calculation
- Change-Point Detection using PELT algorithm
- Stock Price Prediction using Random Forest Regression
- Monte Carlo Simulation for Future Price Forecasting
- Extensive Data Visualization

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib
- ruptures
- scipy

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository to your local machine.
2. Navigate to the directory and run the main script:

```bash
python3 quantAnalysis.py
```

3. The script will generate visualizations which will be displayed on the screen.

## How It Works

### Data Preparation

The code initially generates synthetic stock market data with closing prices and dates, which can be replaced by real-world data.

### Bollinger Bands Calculation

Using a 20-day rolling window, the code calculates the moving average and standard deviation of stock prices. It then computes the upper and lower Bollinger Bands for trading strategy evaluation.

### Change-Point Detection

The PELT (Pruned Exact Linear Time) algorithm identifies significant change-points in stock price trends.

### Random Forest Regression

The code uses Random Forest to predict future stock prices based on features like moving averages and Bollinger Bands.

### Monte Carlo Simulation

A Monte Carlo simulation predicts future stock prices over a short time horizon based on historical volatility and returns.

### Visualization

The code generates a three-panel plot including:

- Stock prices along with Bollinger Bands and change-points
- Monte Carlo simulation of future stock prices
- Histogram of stock prices at the end of the simulation

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
