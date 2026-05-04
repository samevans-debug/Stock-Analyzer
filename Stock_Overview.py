#packages used for this project (you might need to install these packages if missing them)
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
from numpy.ma.core import min_val
from yfinance import Ticker
from nltk.sem.chat80 import borders
from tenacity import retry_unless_exception_type
from datetime import date
from scipy.optimize import minimize


#Page Setup stuff

st.set_page_config(layout="wide")



#API Key needed because the package that we used in class to get the FRED rates is no longer supported...
#my API key is in a .toml file and is NOT uploladed to the github; however, I have added it to the secrets menu on
#streamlit

api_key = st.secrets["FRED_API_KEY"]



st.title(f"Stock Overview", help= "Stock information from YFinance package.")

#Importing data based on Ticker user selects
years = st.sidebar.number_input("Time Horizon", min_value =1, max_value= 200, value= 3, )
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")


try:
    ticker_data = yf.download(ticker, period=f"{years}y", auto_adjust= True)
    ticker_data.columns = ticker_data.columns.get_level_values(0)
except Exception:
    ticker_data = pd.DataFrame()

if not ticker_data.empty:
    # Data for other pages want to calculate  here so that it actually loads before someone leaves the page
    @st.cache_data
    def get_comps(ticker):
        url = f"https://finance.yahoo.com/quote/{ticker}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.text, 'lxml')

        carousel = soup.find('div', class_='carousel-top')

        if carousel is None:
            return []

        comps = []
        for a in carousel.find_all('a', {'data-yga': True}):
            if 'qsp-compare-symbols' in a.get('data-yga', ''):
                label = a.get('aria-label')  # use .get() instead of ['aria-label']
                if label is not None:  # only append if it actually exists
                    comps.append(label)
        return comps[1:-1]


    comps = get_comps(ticker)
    comp_plus_ticker = tuple([ticker] + comps)


    @st.cache_data
    def download_ticker_data(all_tickers):
        tickers_list = list(all_tickers)
        raw_data = yf.download(tickers_list, period='15y', auto_adjust=True)
        return raw_data


    raw_data = download_ticker_data(comp_plus_ticker)
    close_data = raw_data['Close']
    close_data.columns = close_data.columns.get_level_values(0)
    monthly_prices = close_data.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().dropna()

    # Saving the Ticker and Ticker Data so other pages can use it...
    st.session_state["ticker"] = ticker.upper()
    st.session_state["ticker_data"] = ticker_data
    st.session_state['comp_plus_ticker'] = comp_plus_ticker
    st.session_state["comps"] = comps
    st.session_state['raw_data'] = raw_data
    st.session_state['monthly_returns_comps'] = monthly_returns

    #Creatig Moving averages and appending them to the table
    ticker_data['fifty_MA'] = ticker_data["Close"].rolling(50).mean()
    ticker_data['two_hundred_MA'] = ticker_data["Close"].rolling(200).mean()

    #User Toggle for close price vs moving averages and defines ma_col accordingly
    ma_choice = st.sidebar.selectbox("Moving Average Selection", ["50 Day MA", "200 Day MA"])

    if ma_choice == "50 Day MA":
        ma_col = "fifty_MA"
    else:
        ma_col = "two_hundred_MA"

    #the n-year dynamic chart

    if len(ticker_data["Close"])>0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = ticker_data.index,
            y = ticker_data["Close"],
            name = "Close Price",
            line = dict(color = "Grey", width = 1.5)
        ))

        fig.add_trace(go.Scatter(
            x = ticker_data.index,
            y = ticker_data[ma_col],
            name = ma_choice,
            line = dict(color = "Red", width = 1.5)
        ))

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Price (USD)",
            yaxis=dict(autorange=True),
            xaxis=dict(
                rangeslider=dict(visible=False),
                autorange=True
        ))

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)


    #Scraping the Yahoo Finance Website to get summary table
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    page = requests.get(url, headers=headers)

    soup = BeautifulSoup(page.text, 'lxml')

    all_ul_tags = soup.find_all("ul")

    for i, ultag in enumerate(all_ul_tags):
      if "Previous Close" in ultag.text:
        my_ul_tag = ultag


    #my_ul_tag

    all_li_tags = my_ul_tag.find_all('li')


    s_label = []
    s_data = []

    for row in all_li_tags:
      row_span_tags = row.find_all('span')
      s_label.append(row_span_tags[0].text.strip())
      s_data.append(row_span_tags[1].text.strip())

    dictionary_1 = {"Metric": s_label, "": s_data}

    df_summary = (pd.DataFrame(dictionary_1))



    with st.container(border=True):
        st.subheader(f"{ticker.upper()} Summary Table", help= "Summary table information from Yahoo Finance.")
        st.table(df_summary)





else:
    st.warning("_WARNING:_ Stock ticker entered is not available. Please try another ticker before selecting next.")


#getting comps from yahoo finance for other pages






#End of page navigation stuff
if st.button("Next: Portfolio Analysis →"):
    st.switch_page("pages/1_Portfolio_Analysis.py")