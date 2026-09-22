import numpy as np
import pandas as pd


def simulate_portfolio(returns, spy_weight, initial=50000, paths=10000, days=252, seed=42):
    """Correlated multivariate-normal Monte Carlo using historical daily mean/covariance."""
    rng = np.random.default_rng(seed)
    mu = returns[["SPY", "QQQ"]].mean().values
    cov = returns[["SPY", "QQQ"]].cov().values
    weights = np.array([spy_weight, 1-spy_weight])
    asset_draws = rng.multivariate_normal(mu, cov, size=(days, paths))
    port_r = asset_draws @ weights
    wealth = initial * np.cumprod(1 + port_r, axis=0)
    wealth = np.vstack([np.full(paths, initial), wealth])
    percentiles = np.percentile(wealth, [5, 25, 50, 75, 95], axis=1).T
    fan = pd.DataFrame(percentiles, columns=["P05", "P25", "P50", "P75", "P95"])
    fan.insert(0, "Day", np.arange(days + 1))
    final = pd.DataFrame({"Path": np.arange(1, paths+1), "Final_Value": wealth[-1]})
    return fan, final
