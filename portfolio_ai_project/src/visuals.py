from pathlib import Path
import matplotlib.pyplot as plt


def save_efficient_frontier(grid, best, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(grid["Annual_Volatility"], grid["Annual_Return"], c=grid["Sharpe_Ratio"], s=28)
    ax.scatter(best["Annual_Volatility"], best["Annual_Return"], marker="*", s=220, label="Selected portfolio")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("SPY/QQQ Risk-Return Allocation Frontier")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Sharpe Ratio")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_fan_chart(fan, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(fan.Day, fan.P05, fan.P95, alpha=.2, label="5th–95th percentile")
    ax.fill_between(fan.Day, fan.P25, fan.P75, alpha=.3, label="25th–75th percentile")
    ax.plot(fan.Day, fan.P50, label="Median")
    ax.set_xlabel("Trading Day"); ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("10,000-Path Monte Carlo Portfolio Fan Chart")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
