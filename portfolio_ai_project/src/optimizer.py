import numpy as np
import pandas as pd
from .risk_metrics import annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, var_cvar


def portfolio_return_series(returns, spy_weight):
    qqq_weight = 1.0 - spy_weight
    return returns["SPY"] * spy_weight + returns["QQQ"] * qqq_weight


def scan_allocations(returns, risk_free_rate=0.04, trading_days=252, step=0.01):
    rows = []
    for w in np.round(np.arange(0, 1 + step / 2, step), 10):
        r = portfolio_return_series(returns, w)
        var95, cvar95 = var_cvar(r)
        rows.append({
            "SPY_Weight": w,
            "QQQ_Weight": 1-w,
            "Annual_Return": annualized_return(r, trading_days),
            "Annual_Volatility": annualized_volatility(r, trading_days),
            "Sharpe_Ratio": sharpe_ratio(r, risk_free_rate, trading_days),
            "Max_Drawdown": max_drawdown(r),
            "VaR_95_Daily": var95,
            "CVaR_95_Daily": cvar95,
        })
    return pd.DataFrame(rows)


def choose_portfolio(grid, max_drawdown_limit=0.30, cvar_limit=0.03):
    feasible = grid[(grid["Max_Drawdown"] >= -abs(max_drawdown_limit)) &
                    (grid["CVaR_95_Daily"] >= -abs(cvar_limit))]
    pool = feasible if not feasible.empty else grid
    return pool.loc[pool["Sharpe_Ratio"].idxmax()].copy()
