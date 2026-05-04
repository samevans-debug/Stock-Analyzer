from math import trunc
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import plotly.graph_objects as go

ticker = st.session_state.get("ticker", None)
monthly_returns_comps = st.session_state.get("monthly_returns_comps", None)
raw_data = st.session_state.get("raw_data", None)

if ticker is None or raw_data is None or monthly_returns_comps is None:
    st.warning("_WARNING:_ Please complete the Stock Overview and Portfolio pages first.")
    st.stop()

st.title(f"VaR Analysis — {ticker}")

comps_tickers = monthly_returns_comps.columns.tolist()
all_tickers = comps_tickers
N = len(all_tickers)
days = 30

dailyprc = raw_data['Close']
dailyprc.columns = dailyprc.columns.get_level_values(0)

lnret = np.log(dailyprc / dailyprc.shift(1))
lnret = lnret.iloc[1:]

threshold = 0.50
min_obs = int(len(lnret) * threshold)
valid_tickers = lnret.columns[lnret.count() >= min_obs].tolist()
dropped = [t for t in all_tickers if t not in valid_tickers]

with st.container(border=True):
    st.subheader("Investment Value")
    portfolio_value = st.number_input(
        "**Input total investment value (USD):**",
        min_value=1,
        max_value=500000000,
        value=1000000,
        help="Please input the total value of the portfolio that you would like to simulate."
    )

if dropped:
    with st.container(border=True):
        st.warning(f"_WARNING:_ The following tickers were excluded due to insufficient trading history (less than {int(threshold*100)}% of the observation period):")
        for t in dropped:
            st.markdown(f"- **{t}**")

st.subheader("Portfolio Weights",
    help="Click EXPAND below to adjust portfolio weights.")

with st.container(border=True):
    with st.expander("Expand to adjust the portfolio weights"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ticker**", help="Comparable stock tickers from Yahoo Finance.")
        with col2:
            st.markdown("**Weight (%)**", help="Weights must sum to 100%.")

        base_weight = round(100 / len(valid_tickers), 1)

        weights = {}
        for i, t in enumerate(valid_tickers):
            col1, col2 = st.columns(2)
            with col1:
                st.write(t)
            with col2:
                if i == 0:
                    default = round(100 - (base_weight * (len(valid_tickers) - 1)), 1)
                else:
                    default = base_weight

                weights[t] = st.number_input(
                    label=t,
                    min_value=0.0,
                    max_value=100.0,
                    value=default,
                    step=0.1,
                    label_visibility="collapsed"
                )

        total_weight = sum(weights.values())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Total**")
        with col2:
            if abs(total_weight - 100.0) < 0.01:
                st.success(f"**{total_weight:.1f}%** ✓")
            else:
                st.error(f"**{total_weight:.1f}%** — must equal 100%")

st.subheader("Adjust VaR Thresholds")
with st.container(border=True):
    with st.expander("Expand to adjust thresholds:"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Thresholds**")
            st.markdown('**Threshold 1**')
            st.markdown('**Threshold 2**')
        with col2:
            st.markdown("**Level (%)**", help="Percent returns below this threshold.")
            threshold_1 = st.number_input(
                label="t1",
                min_value=0.000001,
                max_value=99.0,
                value=1.0,
                label_visibility="collapsed"
            ) / 100
            threshold_2 = st.number_input(
                label="t2",
                min_value=0.000001,
                max_value=99.0,
                value=5.0,
                label_visibility="collapsed"
            ) / 100

threshold_levels = [threshold_1, threshold_2]

if abs(total_weight - 100.0) < 0.01:

    weights_array = np.array([weights[t] / 100 for t in valid_tickers])
    lnret = lnret[valid_tickers].dropna()
    hist_ret_ew = (weights_array * lnret).sum(axis=1)

    hist_mean = hist_ret_ew.mean()
    hist_std = hist_ret_ew.std()

    hist_Nday_ret = hist_ret_ew.rolling(window=days).sum().dropna()
    hist_Nday_ret_dollars = hist_Nday_ret * portfolio_value

    Var_dict = {"Param": [], "Hist": [], "MC Normal": []}

    # parametric
    for i in threshold_levels:
        z = norm.ppf(i)
        Var = (hist_mean * days - abs(z) * hist_std * np.sqrt(days)) * portfolio_value
        Var_dict['Param'].append(Var)

    # historical
    for i in threshold_levels:
        hist_var = np.percentile(hist_Nday_ret_dollars, i * 100)
        Var_dict['Hist'].append(hist_var)

    # monte carlo
    n_simulations = 100000
    simulated_returns = np.random.normal(
        hist_mean * days,
        hist_std * np.sqrt(days),
        n_simulations
    )
    simulated_dollars = simulated_returns * portfolio_value

    for i in threshold_levels:
        mc_var = np.percentile(simulated_dollars, i * 100)
        Var_dict['MC Normal'].append(mc_var)

    fig_param = go.Figure()
    fig_param.add_trace(go.Histogram(
        x=hist_Nday_ret_dollars,
        nbinsx=50,
        name=f"{days}-Day Returns",
        marker_color="steelblue",
        opacity=0.5,
        histnorm="probability density"
    ))
    for cl, var in zip(threshold_levels, Var_dict["Param"]):
        fig_param.add_vline(
            x=var,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"{int(cl * 100)}% VaR: ${abs(var):,.0f}",
            annotation_position="top left"
        )
    fig_param.update_layout(
        template="plotly_dark",
        xaxis_title=f"{days}-Day Portfolio Return ($)",
        yaxis_title="Frequency",
        title="Parametric VaR — Assumes Normal Distribution"
    )
    with st.container(border=True):
        st.subheader("Parametric VaR",
            help="Assumes returns follow a normal distribution. Uses mean and standard deviation to calculate the loss threshold at each confidence level.")
        st.plotly_chart(fig_param, use_container_width=True, config={"displayModeBar": False})

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=hist_Nday_ret_dollars,
        nbinsx=50,
        name=f"{days}-Day Returns",
        marker_color="steelblue",
        opacity=0.5,
        histnorm="probability density"
    ))
    for cl, var in zip(threshold_levels, Var_dict["Hist"]):
        fig_hist.add_vline(
            x=var,
            line_dash="dash",
            line_color="cyan",
            line_width=2,
            annotation_text=f"{int(cl * 100)}% VaR: ${abs(var):,.0f}",
            annotation_position="top left"
        )
    fig_hist.update_layout(
        template="plotly_dark",
        xaxis_title=f"{days}-Day Portfolio Return ($)",
        yaxis_title="Frequency",
        title="Historical VaR — Uses Actual Return Distribution"
    )
    with st.container(border=True):
        st.subheader("Historical VaR",
            help="Makes no assumption about the distribution of returns. Uses actual historical returns to find the loss threshold directly from the data.")
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=simulated_dollars,
        nbinsx=50,
        name="Simulated Returns",
        marker_color="steelblue",
        opacity=0.5,
        histnorm="probability density"
    ))
    for cl, var in zip(threshold_levels, Var_dict["MC Normal"]):
        fig_mc.add_vline(
            x=var,
            line_dash="dash",
            line_color="magenta",
            line_width=2,
            annotation_text=f"{int(cl * 100)}% VaR: ${abs(var):,.0f}",
            annotation_position="top left"
        )
    fig_mc.update_layout(
        template="plotly_dark",
        xaxis_title=f"{days}-Day Portfolio Return ($)",
        yaxis_title="Frequency",
        title=f"Monte Carlo VaR — {n_simulations:,} Simulated Scenarios"
    )
    with st.container(border=True):
        st.subheader("Monte Carlo VaR", help=f"Simulates {n_simulations:,} return scenarios by drawing randomly from a normal distribution fitted to historical mean and standard deviation.")
        st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar": False})

    with st.container(border=True):
        st.subheader("VaR Summary Table", help="All values represent the _maximum_ expected loss at each confidence level over a 30-day horizon _under normal conditions_.")

        summary_data = {
            "Confidence Level": [f"{int(cl * 100)}%" for cl in threshold_levels],
            "Parametric VaR": [f"${abs(v):,.0f}" for v in Var_dict["Param"]],
            "Historical VaR": [f"${abs(v):,.0f}" for v in Var_dict["Hist"]],
            "Monte Carlo VaR": [f"${abs(v):,.0f}" for v in Var_dict["MC Normal"]],
        }

        st.table(pd.DataFrame(summary_data))

        avg_param = np.mean([abs(v) for v in Var_dict["Param"]])
        avg_hist = np.mean([abs(v) for v in Var_dict["Hist"]])
        avg_mc = np.mean([abs(v) for v in Var_dict["MC Normal"]])


        method_name = {
            avg_param: "Parametric",
            avg_hist: "Historical",
            avg_mc: "Monte Carlo"
        }


else:
    st.warning("_WARNING:_ Please adjust weights to total 100% before running analysis.")