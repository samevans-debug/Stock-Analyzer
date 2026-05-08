import streamlit as st

st.title("Project Write-Up")
st.subheader("_Created by Samuel Evans -- Spring 2026_")
with st.container(border=True):
    st.markdown("**Link to Github Repo:** https://github.com/samevans-debug/Stock-Analyzer")

st.subheader("Project Overview")
st.markdown("At the start of this semester, a goal of mine has been to learn to " \
"code Streamlit applications in Python, and this assignment " \
"was an ideal opportunity for me to achieve this goal. " \
"Building off the requirements outlined in the rubric, I utilized " \
"the streamlit package to improve on the stated goals. Leveraging " \
"Streamlit, I was able to turn this assignment into a dynamic user " \
"friendly web application that provides investment insights into any " \
"company of the user’s choice. \n\n Given this was my first time using " \
"Streamlit, building out the application to work the way I envisioned " \
"it led to numerous hiccups; however, thanks to these challenges I " \
"learned a boat load about how Python, and specifically Streamlit, " \
"works. For instance, I discovered the hard way the benefits and the " \
"downsides of how Streamlit handles UI refreshes when users interact " \
"with an input module. In an effort to prevent resource intensive " \
"modules from re-running after every user interaction on the page–as " \
"this was slowing down the UI– I read the Streamlit documentation. " \
"This led me to discover how to utilize the @cache functionality built " \
"into Streamlit. Another newbie challenge I faced during this project " \
"was learning how to correctly format the file structure of my project "
"for it to be uploaded to GitHub and published to Streamlit web hosting "
"(such as figuring out how to securely share my FRED API key to Streamlit).")

st.subheader("Stock Overview Page")
st.markdown("The main objective of this page is to serve as the" \
"initial ingestion point for new users. This page is where users select"\
"the stock ticker and receive a basic overview of that stock," \
"such as a dynamic price history chart and summary statistics"\
"(ex-dividend date, market cap, etc.). The YFinance package is utilized to"\
"pull the stock price data and the summary statistics are scraped from the"\
"stock’s Yahoo Finance page. On the backend, the .py file for this page"\
"also performs some necessary data ingestion and transformation tasks that"\
"are utilized by later pages in the dashboard. I did this to ensure there"\
"was a singular source of truth for the data, and to reduce duplicate code."\
"To pass these variables to the other page’s .py files I utilized"\
"Streamlit’s st.session_state() function. This was super helpful!")

st.subheader("Portfolio Analysis Page")
st.markdown("This page provides users with information about their selected"\
"stock, along with the comparable companies associated with that stock"\
"(also scraped from the Yahoo Finance website). The user is also provided"\
"the option to remove companies that were identified as comparables as they"\
"deem fit. The dashboard then generates and presents a covariance and"\
"correlation matrix between the stock and the selected comparable"\
"companies. The correlation and covariance matrices are calculated"\
"utilizing the .cov() and .corr() functions. Finally, the Portfolio"\
"Analysis page runs Monte Carlo simulation, randomly generating a"\
"user-specified number of potential portfolios composed of the stock"\
"and its comps. These randomly generated portfolios are then used to"\
"construct an efficiency frontier and find an approximation for the"\
"minimum variance portfolio. The risk free rate used in the efficiency"\
"frontier is sourced from the FRED API, and represents the current"\
"de-annualized monthly return for one-month T-bills. The optimal portfolio"\
"tables display the mathematically optimal minimum variance portfolio and"\
"maximum Sharpe ratio portfolio (subject to the maximum single stock"\
"weight restriction, which is user adjustable).")

st.subheader("News Sentiment Analysis Page")
st.markdown("The purpose of this page is to provide the user with insights"\
"into the news coverage of the firm over time. The table with recent"\
"news articles comes from webscraping the FinViz webpage for the"\
"selected ticker. On the backend this dashboard then scrapes the articles"\
"hyperlinked on the FinViz website and performs VADER sentiment analysis"\
"(by sentence) for each article. An average compound score is then"\
"calculated from the individual sentence compound scores, and that"\
"overall average compound score is displayed in the table on the Sentiment"\
"Analysis page. Articles which have paywalls, such as the NYTimes, are"\
"omitted from the sentiment analysis. While this process is cool, and"\
"taught me a lot about how to do webscraping, it does NOT yield reliable"\
"results. VADER sentiment analysis was initially designed to analyze social"\
"media posts, and from my testing, has proven very unreliable at accurately"\
"assessing the overall sentiment of financial articles.")

st.subheader("Value at Risk")
st.markdown("The last page in this dashboard is the Value at Risk page."\
"The purpose of this page is to provide the user with an idea of how"\
"a hypothetical portfolio consisting of their stock, and comparable stocks,"\
"would perform under certain conditions. Utilizing the three major"\
"approaches to calculating value at risk (VaR)—parametric, historicals,"
"and Monte Carlo—users are able to gain insight into the maximum expected"\
"loss at user-selected confidence levels for the portfolio over a 30-day"\
"period, under normal conditions.")

with st.container(border=True):
    st.image("party_George_logo_for_website_fin.png", use_container_width=True)
