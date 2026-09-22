# QuantRisk AI — Portfolio Intelligence & Risk Decision Platform

An end-to-end portfolio analytics application that converts a $50,000 SPY/QQQ allocation question into a reproducible decision workflow: market-data ingestion, quality controls, constrained optimization, benchmark analysis, Monte Carlo simulation, stress testing, walk-forward validation, and explainable risk reporting.

## Why this is more than a dashboard

The project separates the analytics engine from the presentation layer. It scans 101 candidate allocations, applies explicit downside-risk constraints, validates the strategy across rolling out-of-sample windows, runs 10,000 correlated Monte Carlo paths, and exposes assumptions and limitations through a model card and audit trail.

## Interactive application

Six views are included:

1. **Executive Overview** — KPIs, growth comparison, correlation, benchmark scorecard.
2. **Portfolio Lab** — risk/return opportunity set and interactive what-if allocation controls.
3. **Monte Carlo** — 10,000-path fan chart, loss probability, percentile outcomes, distribution.
4. **Stress Testing** — transparent hypothetical market/technology shocks and portfolio P&L.
5. **Model Validation** — data-quality checks, walk-forward testing, limitations/model card.
6. **AI Risk Analyst** — explainable rule-based decision brief and auditable selection logic.

## Architecture

```text
Market Data (yfinance)
        ↓
Data validation + return engineering
        ↓
101-allocation constrained optimizer
        ├── benchmark analytics
        ├── rolling risk
        ├── walk-forward validation
        ├── scenario stress testing
        └── 10,000-path Monte Carlo
        ↓
Streamlit / Plotly decision application
        ↓
Tableau-ready exports + audit outputs
```

## Run locally (Windows)

```powershell
cd portfolio_ai_project
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run_dashboard.bat` after dependencies are installed.

The terminal will display a local address (normally `http://localhost:8501`) and the application should open in your browser.

## Batch analytics pipeline

```powershell
python main.py
```

This produces CSV exports and static artifacts for downstream BI workflows.

## Methodology

Portfolio weights are scanned in 1% increments. Each candidate is evaluated using annualized return, annualized volatility, Sharpe ratio, maximum drawdown, historical 95% VaR, and historical 95% CVaR. The decision rule first filters allocations using the selected downside-risk limits, then selects the highest historical Sharpe ratio among feasible candidates.

Monte Carlo uses the historical daily mean vector and covariance matrix of SPY/QQQ returns and generates correlated daily returns. A fixed random seed supports reproducibility. Walk-forward validation chooses weights only from each historical training window and evaluates them on the subsequent unseen window.

## Governance and limitations

This is an educational decision-support project, not investment advice. Historical performance does not guarantee future results. The Monte Carlo model assumes multivariate-normal daily returns and can understate extreme tail behavior. Taxes, transaction costs, slippage, regime shifts, and changing correlations are not modeled. The stress scenarios are hypothetical sensitivity tests, not predictions.

## Tech stack

Python · pandas · NumPy · yfinance · Streamlit · Plotly · pytest · Tableau-ready CSV

## Interview framing

> I built a decision platform rather than a single dashboard. The pipeline ingests and validates market data, evaluates 101 portfolio allocations against explicit downside-risk controls, performs walk-forward validation, runs 10,000 correlated Monte Carlo simulations, stress-tests the selected portfolio, and exposes the results through an interactive application with an audit trail and model limitations.
