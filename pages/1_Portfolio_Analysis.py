import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from nltk.sem.chat80 import borders
from tenacity import retry_unless_exception_type
from datetime import date
from scipy.optimize import minimize

api_key = st.secrets["FRED_API_KEY"]


# Session State For Transferring info between pages
ticker = st.session_state.get("ticker", None)
data = st.session_state.get("ticker_data", None)
ticker_data = st.session_state.get('comp_plus_ticker', None)
comps = st.session_state.get('comps', None)
raw_data = st.session_state.get('raw_data', None)

# Checks to make sure a stock was indeed selected on the first Stock_Summary Page
if data is None or not comps or ticker == "":
    st.warning("_WARNING:_ Please enter a ticker on the Stock Overview page first. If you entered a ticker on the first page and are getting this message then it means that no comps were capture for this stock, please choose another to continue.")
    st.stop()

st.title(f"Portfolio Optimization - {ticker}", help = "Based on the last 15 years of stock data.")

# ------------------------------------------------------------------------


# calculates the important stuff for the analysis; I dont know how else to put it..
close_data = raw_data['Close']
close_data.columns = close_data.columns.get_level_values(0)
monthly_prices = close_data.resample('ME').last()
monthly_returns = monthly_prices.pct_change().dropna()




with st.container(border=True):
    st.subheader("Comparable Companies", help="Comparable companies populated from Yahoo Finance.")
    st.write(f"Automatically Identified Comparable Companies to {ticker}. Deselect any you want to exclude.")

    # multiselect lets user remove any comps they don't want to include in the portfolio :)
    selected_comps = st.multiselect(
        "Selected Comps",
        options=comps,
        default=comps
    )

#creates to variable for the dataframes so we only include the companies the user wants to see
selected_tickers = [ticker] + selected_comps
filtered_monthly_returns = monthly_returns[selected_tickers]
filtered_monthly_prices = monthly_prices[selected_tickers]

#cause why not let them see the sauce?
with st.container(border=True):
    with st.expander("View Monthly Price Data For Selected Tickers"):
        st.dataframe(filtered_monthly_prices)






# Doing covariance matrix. The formatting for this is beyond me, but hey the data is right!
cov_matrix = filtered_monthly_returns.cov()

with st.container(border=True):
    st.subheader("Covariance Matrix")
    fig_cov = go.Figure(data=go.Heatmap(
        z=cov_matrix.values,
        x=cov_matrix.columns.tolist(),
        y=cov_matrix.columns.tolist(),
        colorscale="RdBu",
        text=cov_matrix.round(6).values,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_cov.update_layout(
        template="plotly_dark",
    )
    st.plotly_chart(fig_cov, use_container_width=True)

#Doing correlation matrix..

corr_matrix = filtered_monthly_returns.corr()

with st.container(border=True):
    st.subheader("Correlation Matrix")
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale="RdBu",
        text=corr_matrix.round(6).values,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(
        template="plotly_dark",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# running monte carlo simulations
#pulling in FRED (using API call) rate but couldn't use the package we used in class cause not supported in this version of python :(

with st.container(border=True):
    st.subheader("Portfolio Simulation Generator")
    n_portfolios = st.slider(label="Select number monte-carlo simulations to run:", min_value=1, max_value=50000, step=1000, value=50000)
    max_weight = st.slider("Select maximum single-stock weight in portfolio:",min_value=round(1/len(selected_tickers),2),max_value=1.0, step=.01, value= 0.3)


def get_risk_free_rate(api_key):
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=DGS1MO" #this is the code for 1-month treasury bills annualized; some code used in class lecture 25
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&sort_order=desc"
        f"&limit=1"
    )
    response = requests.get(url)
    data = response.json()
    rate = float(data['observations'][0]['value'])
    return rate / 100

risk_free_rate = get_risk_free_rate(api_key)

rf_monthly = risk_free_rate/12
rf_annual = risk_free_rate

means = filtered_monthly_returns.mean()
sigmas = filtered_monthly_returns.std()
r_all = means.values
Sigma_all = filtered_monthly_returns.cov().values
n_stocks = len(selected_tickers)

# from how we did it in class

sim_returns = np.zeros(n_portfolios)
sim_risks = np.zeros(n_portfolios)

for i in range(n_portfolios):
    w = np.random.random(n_stocks)
    w = w / w.sum()
    sim_returns[i] = w @ r_all
    sim_risks[i] = np.sqrt(w @ Sigma_all @ w)

def portfolio_variance(w):
    return w @ Sigma_all @ w

w0 = np.ones(n_stocks) / n_stocks

# Step 3: Define constraints
# Weights must sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

bounds = [(0, max_weight)] * len(selected_tickers)


result_minvar = minimize(portfolio_variance, w0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'ftol': 1e-15})

# Extract results
w_minvar = result_minvar.x
ret_minvar = w_minvar @ r_all
risk_minvar = np.sqrt(w_minvar @ Sigma_all @ w_minvar)
sharpe_max = (ret_minvar - rf_monthly) / risk_minvar

sim_minvar_idx = np.argmin(sim_risks)
sim_minvar_ret = sim_returns[sim_minvar_idx]
sim_minvar_risk = sim_risks[sim_minvar_idx]


def neg_sharpe_ratio(w):
    port_return = w @ r_all
    port_risk = np.sqrt(w @ Sigma_all @ w)
    return -(port_return - rf_monthly) / port_risk

# Constraints: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Optimize (bounds set to ensure not shorting or buying on margin)
result_maxsr = minimize(neg_sharpe_ratio, w0, method='SLSQP', bounds=bounds,
                        constraints=constraints, options={'ftol': 1e-15})

# Extract results
w_maxsr = result_maxsr.x
ret_maxsr = w_maxsr @ r_all
risk_maxsr = np.sqrt(w_maxsr @ Sigma_all @ w_maxsr)
sharpe_maxsr = (ret_maxsr - rf_monthly) / risk_maxsr

sim_sharpes = (sim_returns - rf_monthly) / sim_risks
sim_maxsr_idx = np.argmax(sim_sharpes)
sim_maxsr_ret = sim_returns[sim_maxsr_idx]
sim_maxsr_risk = sim_risks[sim_maxsr_idx]
sim_maxsr_sharpe = sim_sharpes[sim_maxsr_idx]


fig = go.Figure()

# Simulated portfolos
fig.add_trace(go.Scatter(
    x=sim_risks.tolist(),
    y=sim_returns.tolist(),
    mode='markers',
    marker=dict(size=3, color='darkgrey', opacity=0.3),
    name='Simulated Portfolios'
))

# Individuat stocks
for i, t in enumerate(selected_tickers):
    fig.add_trace(go.Scatter(
        x=[sigmas[t]],
        y=[means[t]],
        mode='markers+text',
        marker=dict(size=10, line=dict(width=1, color='black')),
        text=[t],
        textposition='top right',
        name=t
    ))

# Min Variance portfolio
fig.add_trace(go.Scatter(
    x=[risk_minvar],
    y=[ret_minvar],
    mode='markers+text',
    marker=dict(size=15, color='red', symbol='star'),
    text=['Min Var'],
    textposition='top right',
    name='Exact Min Variance'
))

# Max Sharpe portfolio
fig.add_trace(go.Scatter(
    x=[risk_maxsr],
    y=[ret_maxsr],
    mode='markers+text',
    marker=dict(size=15, color='gold', symbol='star',
                line=dict(width=1, color='black')),
    text=['Max Sharpe'],
    textposition='top right',
    name='Exact Max Sharpe'
))

# Capital Market
cml_x = np.linspace(0, float(max(sim_risks)), 100)
cml_y = rf_monthly + sharpe_maxsr * cml_x
fig.add_trace(go.Scatter(
    x=cml_x.tolist(),
    y=cml_y.tolist(),
    mode='lines',
    line=dict(color='black', dash='dash', width=1.5),
    name='Capital Market Line'
))

# Risk free rate point (from the FRED API call using 1m treasuries)
fig.add_trace(go.Scatter(
    x=[0],
    y=[rf_monthly],
    mode='markers+text',
    marker=dict(size=8, color='black'),
    text=[f'Risk-Free ({rf_monthly:.1%} monthly)'],
    textposition='middle right',
    name='Risk-Free Rate'
))

fig.update_layout(
    template='plotly_dark',
    xaxis_title='Risk (Std Dev of Monthly Returns)',
    yaxis_title='Expected Return (Mean Monthly Return)',
)

with st.container(border=True):
    st.subheader(f"Efficiency Frontier Generated from {n_portfolios} Simulations", help="Uses monte-carlo simulation. Risk free rate is from the current one month treasury bill rate (Source: FRED).")
    st.plotly_chart(fig, use_container_width=True)


with st.container(border=True):
    st.subheader("Optimal Portfolios Overview")
    view = st.radio("Select Return Period", ["Monthly", "Annual"], horizontal=True)
    if view == "Annual":
        display_ret_maxsr = (1 + ret_maxsr) ** 12 - 1
        display_risk_maxsr = risk_maxsr * np.sqrt(12)
        display_ret_minvar = (1 + ret_minvar) ** 12 - 1
        display_risk_minvar = risk_minvar * np.sqrt(12)
    else:
        display_ret_maxsr = ret_maxsr
        display_risk_maxsr = risk_maxsr
        display_ret_minvar = ret_minvar
        display_risk_minvar = risk_minvar

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Min Variance Portfolio")
        weights_minvar = pd.DataFrame({
            'Ticker': selected_tickers,
            'Weight': [f"{w:.2%}" for w in w_minvar]
        })
        st.dataframe(weights_minvar, use_container_width=True)
        st.metric("Expected Return", f"{display_ret_minvar:.2%}")
        st.metric("Risk (Std Dev)", f"{display_risk_minvar:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_max:.4f}")


    with col2:
        st.subheader("Global Max Sharpe Portfolio")
        weights_maxsr = pd.DataFrame({
            'Ticker': selected_tickers,
            'Weight': [f"{w:.2%}" for w in w_maxsr]
        })
        st.dataframe(weights_maxsr, use_container_width=True)
        st.metric("Expected Return", f"{display_ret_maxsr:.2%}")
        st.metric("Risk (Std Dev)", f"{display_risk_maxsr:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_maxsr:.4f}")


#Saving the comps ticker list to session state fot the value at risk page
st.session_state["monthly_returns_comps"] = monthly_returns



#End of Page navigation stuff
col1, col2 = st.columns(2)
with col1:
    if st.button("← Back: Stock Overview"):
        st.switch_page("app.py")
with col2:
    if st.button("Next: News Sentiment →"):
        st.switch_page("pages/2_News_Sentiment_Analysis.py")

