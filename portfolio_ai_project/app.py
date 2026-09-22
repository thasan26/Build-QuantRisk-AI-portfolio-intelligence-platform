import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data_pipeline import download_prices, daily_returns
from src.optimizer import scan_allocations, choose_portfolio, portfolio_return_series
from src.monte_carlo import simulate_portfolio
from src.advanced_analytics import performance_table, rolling_metrics, stress_test, walk_forward, data_quality

ROOT=Path(__file__).parent
CFG=json.loads((ROOT/'config.json').read_text())
st.set_page_config(page_title='QuantRisk AI', page_icon='📈', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.4rem}.metric-card{border:1px solid #30363d;border-radius:12px;padding:12px}.small{opacity:.72;font-size:.85rem}</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(years):
    p=download_prices(['SPY','QQQ'],period=f'{years}y')
    return p,daily_returns(p)

st.sidebar.title('QuantRisk AI')
st.sidebar.caption('Portfolio Intelligence & Risk Decision Platform')
years=st.sidebar.slider('Historical lookback (years)',2,10,5)
initial=st.sidebar.number_input('Initial investment ($)',10000,1000000,50000,5000)
risk_profile=st.sidebar.selectbox('Risk profile',['Conservative','Moderate','Aggressive'])
limits={'Conservative':(.20,.020),'Moderate':(.30,.030),'Aggressive':(.40,.040)}
max_dd,cvar_lim=limits[risk_profile]
page=st.sidebar.radio('Navigate',['Executive Overview','Portfolio Lab','Monte Carlo','Stress Testing','Model Validation','AI Risk Analyst'])
st.sidebar.divider(); st.sidebar.caption('Educational analytics only — not investment advice.')

try: prices,returns=load_data(years)
except Exception as e:
    st.error(f'Market data could not be loaded: {e}'); st.stop()
grid=scan_allocations(returns,CFG['risk_free_rate'],252,.01)
best=choose_portfolio(grid,max_dd,cvar_lim)
w=float(best['SPY_Weight']); r=portfolio_return_series(returns,w)

if page=='Executive Overview':
    st.title('QuantRisk AI — Executive Command Center')
    st.caption('End-to-end market data → risk optimization → simulation → decision intelligence')
    a,b,c,d,e=st.columns(5)
    a.metric('Recommended SPY',f'{w:.0%}'); b.metric('Recommended QQQ',f'{1-w:.0%}')
    c.metric('Sharpe',f"{best['Sharpe_Ratio']:.2f}"); d.metric('Annual Return',f"{best['Annual_Return']:.1%}"); e.metric('Max Drawdown',f"{best['Max_Drawdown']:.1%}")
    wealth=(1+returns.assign(Selected=r)[['SPY','QQQ','Selected']]).cumprod()*initial
    fig=px.line(wealth,title=f'Growth of ${initial:,.0f}: Selected Portfolio vs Benchmarks')
    st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        corr=returns[['SPY','QQQ']].corr(); st.plotly_chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,title='Asset Correlation'),use_container_width=True)
    with right:
        perf=performance_table(returns,w,initial,CFG['risk_free_rate'])
        show=perf.copy();
        for col in ['Annual Return','Volatility','Max Drawdown','Daily VaR 95%','Daily CVaR 95%']: show[col]=show[col].map(lambda x:f'{x:.2%}')
        show['Sharpe']=show['Sharpe'].map(lambda x:f'{x:.2f}'); show['Ending Value']=show['Ending Value'].map(lambda x:f'${x:,.0f}')
        st.subheader('Benchmark Scorecard'); st.dataframe(show,hide_index=True,use_container_width=True)

elif page=='Portfolio Lab':
    st.title('Portfolio Optimization Lab')
    st.write('Explore the complete 101-allocation opportunity set. The highlighted point maximizes historical Sharpe subject to the selected downside-risk limits.')
    fig=px.scatter(grid,x='Annual_Volatility',y='Annual_Return',color='Sharpe_Ratio',hover_data=['SPY_Weight','QQQ_Weight','Max_Drawdown','CVaR_95_Daily'],title='Risk–Return Opportunity Set')
    fig.add_trace(go.Scatter(x=[best['Annual_Volatility']],y=[best['Annual_Return']],mode='markers',marker=dict(size=18,symbol='star'),name='Selected'))
    st.plotly_chart(fig,use_container_width=True)
    custom=st.slider('What-if SPY allocation',0,100,int(round(w*100)))/100
    cr=portfolio_return_series(returns,custom); comp=performance_table(returns,custom,initial,CFG['risk_free_rate']).iloc[0]
    a,b,c,d=st.columns(4); a.metric('SPY / QQQ',f'{custom:.0%} / {1-custom:.0%}'); b.metric('Annual Return',f"{comp['Annual Return']:.1%}"); c.metric('Volatility',f"{comp['Volatility']:.1%}"); d.metric('Sharpe',f"{comp['Sharpe']:.2f}")
    roll=rolling_metrics(cr)
    st.plotly_chart(px.line(roll[['Rolling Volatility','Drawdown']],title='Rolling Risk & Drawdown'),use_container_width=True)

elif page=='Monte Carlo':
    st.title('10,000-Path Monte Carlo Risk Lab')
    horizon=st.slider('Simulation horizon (trading days)',63,756,252,63)
    with st.spinner('Running correlated simulations...'):
        fan,final=simulate_portfolio(returns,w,initial,10000,horizon,42)
    fig=go.Figure(); x=fan['Day']
    fig.add_trace(go.Scatter(x=x,y=fan.P95,line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=x,y=fan.P05,fill='tonexty',name='5–95% range',line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x,y=fan.P75,line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=x,y=fan.P25,fill='tonexty',name='25–75% range',line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x,y=fan.P50,name='Median',line=dict(width=3))); fig.update_layout(title='Portfolio Value Fan Chart',yaxis_title='Portfolio Value ($)')
    st.plotly_chart(fig,use_container_width=True)
    vals=final.Final_Value; a,b,c,d=st.columns(4); a.metric('Median Final',f'${vals.median():,.0f}'); b.metric('5th Percentile',f'${vals.quantile(.05):,.0f}'); c.metric('95th Percentile',f'${vals.quantile(.95):,.0f}'); d.metric('Probability of Loss',f'{(vals<initial).mean():.1%}')
    st.plotly_chart(px.histogram(final,x='Final_Value',nbins=70,title='Distribution of Simulated Ending Values'),use_container_width=True)

elif page=='Stress Testing':
    st.title('Scenario Stress Testing')
    stress=stress_test(returns,w,initial)
    fig=px.bar(stress,x='Scenario',y='P&L',text_auto=',.0f',title='Hypothetical Portfolio P&L Under Deterministic Shocks')
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(stress.style.format({'SPY Shock':'{:.1%}','QQQ Shock':'{:.1%}','Portfolio Return':'{:.1%}','P&L':'${:,.0f}','Portfolio Value':'${:,.0f}'}),use_container_width=True,hide_index=True)
    st.info('These scenarios are transparent hypothetical shocks for sensitivity analysis; they are not forecasts.')

elif page=='Model Validation':
    st.title('Model Validation & Data Governance')
    st.subheader('Data Quality Controls'); st.dataframe(data_quality(prices),hide_index=True,use_container_width=True)
    wf=walk_forward(returns,CFG['risk_free_rate'])
    if wf.empty: st.warning('Increase lookback period for walk-forward validation.')
    else:
        a,b,c=st.columns(3); a.metric('Validation Windows',len(wf)); b.metric('Positive Forward Windows',f"{(wf['Forward Return']>0).mean():.1%}"); c.metric('Avg Forward Return',f"{wf['Forward Return'].mean():.1%}")
        st.plotly_chart(px.bar(wf,x='Date',y='Forward Return',color='SPY Weight',title='Walk-Forward Out-of-Sample Returns'),use_container_width=True)
        st.dataframe(wf.style.format({'SPY Weight':'{:.0%}','QQQ Weight':'{:.0%}','Forward Return':'{:.2%}','Forward Volatility':'{:.2%}'}),use_container_width=True,hide_index=True)
    st.subheader('Model Card')
    st.markdown('**Objective:** maximize historical Sharpe under drawdown/CVaR constraints.  **Inputs:** adjusted daily SPY/QQQ prices.  **Validation:** rolling out-of-sample windows.  **Limitations:** historical relationships may not persist; Gaussian Monte Carlo can understate tail risk; transaction costs and taxes are excluded.')

else:
    st.title('AI Risk Analyst')
    st.caption('Explainable, rule-based decision narrative — no external LLM or API key required.')
    var=float(best['VaR_95_Daily']); cvar=float(best['CVaR_95_Daily'])
    st.subheader('Decision Brief')
    st.write(f"For the **{risk_profile.lower()}** profile, the constraint-aware optimizer selected **{w:.0%} SPY / {1-w:.0%} QQQ** from 101 candidate allocations. Over the selected {years}-year historical window, this allocation produced an annualized return of **{best['Annual_Return']:.1%}**, annualized volatility of **{best['Annual_Volatility']:.1%}**, and Sharpe ratio of **{best['Sharpe_Ratio']:.2f}**.")
    st.write(f"The observed maximum drawdown was **{best['Max_Drawdown']:.1%}**. Historical one-day 95% VaR was **{var:.2%}**, while CVaR was **{cvar:.2%}**. The optimizer only treats allocations inside the configured drawdown and CVaR limits as feasible before maximizing Sharpe.")
    st.warning('Decision support, not investment advice. Results depend on historical data and modeling assumptions.')
    st.subheader('Audit Trail')
    st.code(f'Profile={risk_profile}\nDrawdown limit={max_dd:.0%}\nCVaR limit={cvar_lim:.1%}\nCandidates=101\nObjective=max Sharpe among feasible portfolios\nSelected SPY={w:.0%}\nSelected QQQ={1-w:.0%}',language='text')
