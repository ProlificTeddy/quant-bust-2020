import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Define a simple Statistical Arbitrage strategy
class StatisticalArbitrage:
    def __init__(self, lookback=20, threshold=1.5):
        self.lookback = lookback
        self.threshold = threshold

    def z_score(self, series):
        mean = np.mean(series)
        std = np.std(series)
        return (series[-1] - mean) / std if std > 0 else 0

    def generate_signals(self, price_series):
        signals = []
        for i in range(len(price_series)):
            if i < self.lookback:
                signals.append(0)  # Not enough data for lookback period
                continue
            z = self.z_score(price_series[i - self.lookback:i])
            if z > self.threshold:
                signals.append(-1)  # Sell signal
            elif z < -self.threshold:
                signals.append(1)  # Buy signal
            else:
                signals.append(0)  # No action
        return np.array(signals)

    def backtest(self, price_series):
        signals = self.generate_signals(price_series)
        returns = np.diff(price_series) / price_series[:-1]
        strategy_returns = signals[:-1] * returns
        cumulative_returns = np.cumsum(strategy_returns)
        return cumulative_returns, signals

if __name__ == '__main__':
    # Generate dummy data for testing
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, 252)) + 100  # Simulated price series

    # Initialize and test the Statistical Arbitrage strategy
    lookback_period = 20
    z_threshold = 1.5
    stat_arb = StatisticalArbitrage(lookback=lookback_period, threshold=z_threshold)

    cumulative_returns, signals = stat_arb.backtest(prices)

    # Print results
    print("Cumulative Returns:", cumulative_returns)
    print("Signals:", signals)

    # Plot results for visualization
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price")
    plt.plot(cumulative_returns, label="Cumulative Returns", linestyle="--")
    plt.legend()
    plt.title("Statistical Arbitrage Backtest")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()