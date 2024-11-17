import os
import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def fetch_finnhub_news(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch stock-related news articles using Finnhub API.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        days_back (int): Number of days back for news.

    Returns:
        pd.DataFrame: DataFrame containing news articles.
    """
    try:
        # Calculate date range for news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker.upper(),
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse the response and convert to DataFrame
        news_data = response.json()
        if not news_data:
            return pd.DataFrame()

        news_df = pd.DataFrame(news_data)

        # Safeguard against invalid timestamps
        if "datetime" in news_df.columns:
            news_df = news_df[pd.to_numeric(news_df["datetime"], errors="coerce").notnull()]  # Remove invalid timestamps
            news_df["datetime"] = pd.to_datetime(news_df["datetime"], unit="s", errors="coerce")  # Convert valid timestamps
            news_df = news_df.dropna(subset=["datetime"])  # Drop rows where datetime conversion failed

        # Select and rename relevant columns
        news_df = news_df[["headline", "url", "source", "datetime"]]
        news_df.columns = ["Title", "Link", "Publisher", "Date"]

        return news_df
    except Exception as e:
        st.error(f"Error fetching news for {ticker} from Finnhub: {str(e)}")
        return pd.DataFrame()

def display_stock_info(ticker: str):
    """Display detailed stock information using Finnhub."""
    try:
        url = f"https://finnhub.io/api/v1/quote"
        params = {"symbol": ticker.upper(), "token": FINNHUB_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()

        stock_data = response.json()
        
        # Safely retrieve values, defaulting to 'N/A' if not found
        current_price = stock_data.get('c')
        percent_change = stock_data.get('dp')
        day_high = stock_data.get('h')
        day_low = stock_data.get('l')
        prev_close = stock_data.get('pc')

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Current Price",
                value=f"${current_price:,.2f}" if current_price is not None else "N/A",
                delta=f"{percent_change:.2f}%" if percent_change is not None else "N/A"
            )

        with col2:
            st.metric(
                label="Day High",
                value=f"${day_high:,.2f}" if day_high is not None else "N/A"
            )
            st.metric(
                label="Day Low",
                value=f"${day_low:,.2f}" if day_low is not None else "N/A"
            )
            st.metric(
                label="Previous Close",
                value=f"${prev_close:,.2f}" if prev_close is not None else "N/A"
            )
    except Exception as e:
        st.error(f"Error fetching stock info for {ticker}: {str(e)}")

def stock_news_page():
    """Main function to display the Stock News Research page."""
    # Initialize session state if it doesn't exist
    if "research_urls" not in st.session_state:
        st.session_state.research_urls = set()
    if "manual_urls" not in st.session_state:
        st.session_state.manual_urls = [""] * 3

    st.title("📰 Stock News Research")

    # Sidebar inputs
    st.sidebar.title("⚙️ Settings")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL").upper()
    days_back = st.sidebar.slider("Days of News", min_value=1, max_value=90, value=30)

    if ticker:
        try:
            # Display stock information
            display_stock_info(ticker)

            # Fetch and display news
            st.subheader(f"Recent News for {ticker}")
            news_df = fetch_finnhub_news(ticker, days_back)

            if news_df.empty:
                st.warning(f"No recent news found for {ticker}")
            else:
                # Display news articles
                for idx, article in news_df.iterrows():
                    with st.expander(f"{article['Date'].strftime('%Y-%m-%d')}: {article['Title']}"):
                        st.write(f"**Publisher:** {article['Publisher']}")
                        st.write(f"**Link:** [{article['Title']}]({article['Link']})")

                        # Modified button implementation
                        if st.button(f"Use in Research Tool", key=f"use_button_{idx}"):
                            # Find the first empty manual URL slot
                            empty_slot = -1
                            for i, url in enumerate(st.session_state.manual_urls):
                                if not url:
                                    empty_slot = i
                                    break

                            if empty_slot != -1:
                                # Add to first empty slot
                                st.session_state.manual_urls[empty_slot] = article["Link"]
                                st.success(f"Article added to URL slot {empty_slot + 1}!")
                            else:
                                # Add to research_urls if no empty slots
                                st.session_state.research_urls.add(article["Link"])
                                st.success("All URL slots full. Article added to additional research URLs!")

                # Download option
                csv = news_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download News Data",
                    data=csv,
                    file_name=f"{ticker}_news.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing ticker {ticker}: {str(e)}")

if __name__ == "__main__":
    stock_news_page()