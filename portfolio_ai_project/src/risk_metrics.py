import numpy as np


def annualized_return(r, trading_days=252):
    return (1 + r).prod() ** (trading_days / len(r)) - 1


def annualized_volatility(r, trading_days=252):
    return r.std(ddof=1) * np.sqrt(trading_days)


def sharpe_ratio(r, risk_free_rate=0.04, trading_days=252):
    vol = annualized_volatility(r, trading_days)
    return np.nan if vol == 0 else (annualized_return(r, trading_days) - risk_free_rate) / vol


def max_drawdown(r):
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def var_cvar(r, alpha=0.05):
    var = float(np.quantile(r, alpha))
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar
