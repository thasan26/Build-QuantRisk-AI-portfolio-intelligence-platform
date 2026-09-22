import numpy as np
import pandas as pd
from .optimizer import portfolio_return_series
from .risk_metrics import annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, var_cvar


def performance_table(returns, spy_weight, initial=50000, rf=0.04):
    series = {
        'Selected Portfolio': portfolio_return_series(returns, spy_weight),
        '50/50 Portfolio': portfolio_return_series(returns, .5),
        'SPY': returns['SPY'], 'QQQ': returns['QQQ']
    }
    rows=[]
    for name,r in series.items():
        v,c=var_cvar(r)
        rows.append({'Portfolio':name,'Annual Return':annualized_return(r),'Volatility':annualized_volatility(r),
                     'Sharpe':sharpe_ratio(r,rf),'Max Drawdown':max_drawdown(r),'Daily VaR 95%':v,'Daily CVaR 95%':c,
                     'Ending Value':initial*(1+r).prod()})
    return pd.DataFrame(rows)


def rolling_metrics(r, window=63, rf=.04):
    wealth=(1+r).cumprod()
    out=pd.DataFrame(index=r.index)
    out['Rolling Return']=(1+r).rolling(window).apply(np.prod, raw=True)-1
    out['Rolling Volatility']=r.rolling(window).std()*np.sqrt(252)
    out['Rolling Sharpe']=(r.rolling(window).mean()*252-rf)/(r.rolling(window).std()*np.sqrt(252))
    out['Drawdown']=wealth/wealth.cummax()-1
    return out


def stress_test(returns, spy_weight, initial=50000):
    # Transparent deterministic shocks; scenario analysis, not forecasts.
    scenarios={
        'Market correction':(-.15,-.20),
        'Tech shock':(-.10,-.30),
        'Broad selloff':(-.25,-.25),
        'Risk-on rally':(.12,.20),
        'Mild recession':(-.12,-.16),
    }
    rows=[]
    for name,(spy,qqq) in scenarios.items():
        pr=spy_weight*spy+(1-spy_weight)*qqq
        rows.append({'Scenario':name,'SPY Shock':spy,'QQQ Shock':qqq,'Portfolio Return':pr,
                     'P&L':initial*pr,'Portfolio Value':initial*(1+pr)})
    return pd.DataFrame(rows)


def walk_forward(returns, rf=.04, train_days=504, rebalance_days=63):
    records=[]
    for start in range(train_days, len(returns)-rebalance_days+1, rebalance_days):
        train=returns.iloc[start-train_days:start]
        test=returns.iloc[start:start+rebalance_days]
        candidates=[]
        for w in np.linspace(0,1,101):
            r=portfolio_return_series(train,w)
            candidates.append((sharpe_ratio(r,rf),w))
        _,w=max(candidates,key=lambda x: -np.inf if np.isnan(x[0]) else x[0])
        tr=portfolio_return_series(test,w)
        records.append({'Date':test.index[-1],'SPY Weight':w,'QQQ Weight':1-w,
                        'Forward Return':(1+tr).prod()-1,'Forward Volatility':tr.std()*np.sqrt(252)})
    return pd.DataFrame(records)


def data_quality(prices):
    return pd.DataFrame([
        {'Check':'Rows','Value':len(prices),'Status':'PASS' if len(prices)>500 else 'WARN'},
        {'Check':'Missing values','Value':int(prices.isna().sum().sum()),'Status':'PASS' if not prices.isna().any().any() else 'FAIL'},
        {'Check':'Duplicate dates','Value':int(prices.index.duplicated().sum()),'Status':'PASS' if not prices.index.duplicated().any() else 'FAIL'},
        {'Check':'Non-positive prices','Value':int((prices<=0).sum().sum()),'Status':'PASS' if not (prices<=0).any().any() else 'FAIL'},
        {'Check':'Assets','Value':', '.join(prices.columns),'Status':'PASS'},
    ])
