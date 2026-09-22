import pandas as pd
from src.risk_metrics import max_drawdown, var_cvar
from src.optimizer import portfolio_return_series

def test_no_drawdown_positive_returns():
    r = pd.Series([0.01, 0.02, 0.005])
    assert max_drawdown(r) == 0.0

def test_weights():
    df = pd.DataFrame({'SPY':[.01], 'QQQ':[.03]})
    assert abs(portfolio_return_series(df, .5).iloc[0] - .02) < 1e-12

def test_var_cvar_order():
    r = pd.Series([-.10, -.05, 0, .02, .03])
    var, cvar = var_cvar(r)
    assert cvar <= var
