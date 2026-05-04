import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests import Timeout, RequestException
from trafilatura import extract
import pandas as pd
import plotly.graph_objects as go
import nltk
nltk.download('vader_lexicon', quiet=True) #this is for reading the articles' text


# Page setup instuctions
api_key = st.secrets["FRED_API_KEY"]


#pulling in ticker from input screen on the Stock Overview page
ticker = st.session_state.get("ticker", None)

if ticker is None or ticker == "":
    st.warning("_WARNING:_ Please enter a ticker on the Stock Overview page first.")
    st.stop()

st.title(f"News Sentiment Analysis — {ticker}")



# gets the articles listed on the finviz website, opens those articles, calcs the compound vader score, then adds all this info to a table
@st.cache_data
def get_sentiment(ticker):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
    }

    # dynamic ticker in URL
    html_data = requests.get(
        f'https://finviz.com/quote.ashx?t={ticker}',
        headers=headers
    ).text

    jsoup = BeautifulSoup(html_data, 'lxml')
    news_table = jsoup.find('table', id='news-table')

    if news_table is None:
        return pd.DataFrame()

    page_url = "https://finviz.com"
    rows = news_table.find_all('tr')

    Date = []
    Time = []
    Title = []
    urls = []
    current_date = ""

    for row in rows:
        date_cell = row.find('td', align='right')
        link_title = row.find('a', class_="tab-link-news")
        link = row.find('a')

        if not date_cell or not link_title or not link:
            continue

        clean_url = urljoin(page_url, link['href'])
        urls.append(clean_url)

        raw_dt = date_cell.text.strip().split(" ")
        if len(raw_dt) == 2:
            current_date = raw_dt[0]
            Time.append(raw_dt[1])
            Date.append(current_date)
        else:
            Time.append(raw_dt[0])
            Date.append(current_date)

        Title.append(link_title.text.strip())


    sentiment = SentimentIntensityAnalyzer()
    compound_scores = []
#starts to actually take the urls which were added to the dataframe open them extract the text and run it through vader
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                article_text = extract(response.text)
                if article_text:
                    vs = sentiment.polarity_scores(article_text)
                    compound_scores.append(vs['compound'])
                else:
                    compound_scores.append(None)
            else:
                compound_scores.append(None)
        except Timeout:
            compound_scores.append(None)
        except RequestException:
            compound_scores.append(None)
#coverts the lists that everything was added to into a dictionary and then from the dict to a pandas dataframe
    df = pd.DataFrame({
        "Date": Date,
        "Time": Time,
        "Headline": Title,
        "Compound Score": compound_scores,
        "URL": urls
    })

    return df #function does it all! returns completed data frame with compound sentiment scores for the articles..

#this calls the get_sentiment function using the ticker user input on the first screen and notifies user that the process
#is happening # I read the 'design of everyday things book' and the importance of giving users input feedback...
with st.spinner("Scraping news and analyzing sentiment... this may take a minute (like literally a minute..)"):
    df_sentiment = get_sentiment(ticker)


#provides the user with an error message if for some reason the finviz function didnt work
if df_sentiment.empty:
    st.error("Could not retrieve news. FinViz may have blocked the request.")
    st.stop()

# okay so this ONLY drops columns where the observation has an na in the 'compound score' column.' it ignores nas in
#other columns

df_display = df_sentiment.dropna(subset=["Compound Score"])

#so this was kinda annoying and just had claude do it but essentially it formats the article column to just be the
#clickable url to the article
st.dataframe(
    df_display[["Date", "Time", "Headline", "Compound Score", "URL"]],
    column_config={
        "URL": st.column_config.LinkColumn(
            "Article",
            display_text="Link"
        )
    },
    use_container_width=True
)


avg_score = df_sentiment["Compound Score"].mean()


#displays in a separate container the overall mean sentiment score along with if it was "average", "good", etc..
with st.container(border=True):
    st.subheader("Overall Sentiment", help="Overall sentiment score is calculated by averaging the VADER compound sentiment scores for the articles included on the finviz page for the ticker you selected. Articles which have paywalls, such at the NYTimes, are omitted from the sentiment analysis. The compound score is a 'normalized, weighted composite score between -1 (extremely negative) and +1 (extremely positive)'. learn more about vader https://deepwiki.com/cjhutto/vaderSentiment/4.3-interpreting-results")
    st.metric("Average Compound Score", f"{avg_score:.3f}")
    if avg_score >= 0.05:
        st.success("Overall sentiment is POSITIVE")
    elif avg_score <= -0.05:
        st.error("Overall sentiment is NEGATIVE")
    else:
        st.warning("Overall sentiment is NEUTRAL")

with st.container(border=True):
    st.subheader("Sentiment Over Time")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sentiment["Date"],
        y=df_sentiment["Compound Score"],
        marker_color=df_sentiment["Compound Score"].apply(
            lambda x: "green" if x > 0 else "red"
        )
    ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Compound Score",
        title=f"News Sentiment for {ticker}"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

#End of page navigation stuff

col1, col2 = st.columns(2)
with col1:
    if st.button("← Back: Portfolio Analysis"):
        st.switch_page("pages/1_Portfolio_Analysis.py")
with col2:
    if st.button("Next: Value at Risk →"):
        st.switch_page("pages/3_Value_At_Risk.py")