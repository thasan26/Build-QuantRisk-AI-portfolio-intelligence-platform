import json
from pathlib import Path
import pandas as pd
from src.data_pipeline import download_prices, daily_returns
from src.optimizer import scan_allocations, choose_portfolio, portfolio_return_series
from src.monte_carlo import simulate_portfolio
from src.visuals import save_efficient_frontier, save_fan_chart

ROOT = Path(__file__).parent
cfg = json.loads((ROOT / "config.json").read_text())

prices = download_prices(cfg["tickers"], period=f'{cfg["lookback_years"]}y', output_path=ROOT/'data/processed/prices.csv')
returns = daily_returns(prices)
returns.to_csv(ROOT/'data/processed/daily_returns.csv', index_label='Date')

grid = scan_allocations(returns, cfg['risk_free_rate'], cfg['trading_days'], cfg['weight_step'])
best = choose_portfolio(grid, cfg['max_drawdown_limit'], cfg['cvar_limit'])
grid.to_csv(ROOT/'tableau_exports/allocation_metrics.csv', index=False)
pd.DataFrame([best]).to_csv(ROOT/'tableau_exports/selected_portfolio.csv', index=False)

# Long-format historical dataset for Tableau
long_prices = prices.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Adjusted_Close')
long_prices.to_csv(ROOT/'tableau_exports/historical_prices.csv', index=False)

fan, final = simulate_portfolio(returns, float(best['SPY_Weight']), cfg['initial_investment'], cfg['monte_carlo_paths'], cfg['monte_carlo_days'], cfg['random_seed'])
fan.to_csv(ROOT/'tableau_exports/monte_carlo_fan.csv', index=False)
final.to_csv(ROOT/'tableau_exports/monte_carlo_final_values.csv', index=False)

# Risk-profile recommendations from same tested allocation universe
profiles = []
for name, dd, cvar in [('Conservative', .20, .02), ('Moderate', .30, .03), ('Aggressive', .45, .05)]:
    row = choose_portfolio(grid, dd, cvar)
    profiles.append({'Risk_Profile': name, **row.to_dict()})
pd.DataFrame(profiles).to_csv(ROOT/'tableau_exports/risk_profile_recommendations.csv', index=False)

save_efficient_frontier(grid, best, ROOT/'outputs/figures/allocation_frontier.png')
save_fan_chart(fan, ROOT/'outputs/figures/monte_carlo_fan.png')

summary = f'''Selected portfolio\nSPY: {best['SPY_Weight']:.0%}\nQQQ: {best['QQQ_Weight']:.0%}\nAnnualized return: {best['Annual_Return']:.2%}\nAnnualized volatility: {best['Annual_Volatility']:.2%}\nSharpe ratio: {best['Sharpe_Ratio']:.3f}\nMax drawdown: {best['Max_Drawdown']:.2%}\nDaily 95% CVaR: {best['CVaR_95_Daily']:.2%}\nMonte Carlo median final value: ${final['Final_Value'].median():,.0f}\n'''
(ROOT/'outputs/summary.txt').write_text(summary)
print(summary)
