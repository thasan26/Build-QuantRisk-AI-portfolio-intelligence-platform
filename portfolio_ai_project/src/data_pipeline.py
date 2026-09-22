from pathlib import Path
import pandas as pd
import yfinance as yf


def download_prices(tickers, period="5y", output_path=None):
    """Download adjusted daily close prices and run basic data-quality checks."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No market data returned. Check internet connection/tickers.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.dropna(how="all").ffill().dropna()
    missing = close.isna().sum().sum()
    if missing:
        raise ValueError(f"Price data still contains {missing} missing values after cleaning.")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        close.to_csv(output_path, index_label="Date")
    return close


def daily_returns(prices):
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough observations to calculate returns.")
    return returns
