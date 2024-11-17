import os
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# First, create a new file called pages/stock_news.py
# This will automatically create a new page in your Streamlit app

def get_stock_news(ticker: str, days_back: int = 30) -> pd.DataFrame:
    
    
    """
    Fetch news for a given stock ticker using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        days_back (int): Number of days of news to fetch
        
    Returns:
        pd.DataFrame: DataFrame containing news articles
    """
    
    try: 
        
        # get stock info
        stock = yf.Ticker(ticker)
        
        # fetch stock news
        news = stock.news
        
        if not news:
            return pd.DataFrame()
        
        # convert to DataFrame
        news_df = pd.DataFrame(news)
        
        # convert timestamp to datestamp
        news_df['providerPublishTime'] = pd.to_datetime(news_df['providerPublishTime'], unit = 's')
        
        # filter out for recent news
        cutoff_date = datetime.now() - timedelta(days=days_back)
        news_df = news_df[['title', 'link', 'publisher', 'providerPublishTime', 'type']]
        news_df.columns = ['Title', 'Link', 'Publisher', 'Date', 'Type']
        
        return news_df
    
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return pd.DataFrame()
    
def display_stock_info(ticker: str):
    """Display basic stock information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    col1, col2, = st.columns(2)
    
    with col1:
        st.metric(
            label= "Current Price",
            value= f"${info.get('currentPrice', 'N/A'):,.2f}",
            delta=f"{info.get('regularMarketChangePercent', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            label="Volume",
            value=f"{info.get('volume', 'N/A'):,}"
        )
        
def stock_news_page():
    # Initialize session state if it doesn't exist
    if 'research_urls' not in st.session_state:
        st.session_state.research_urls = set()
    if 'manual_urls' not in st.session_state:
        st.session_state.manual_urls = [''] * 3
        
    st.title("📰 Stock News Research")
    
    # Sidebar inputs
    st.sidebar.title("⚙️ Settings")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL").upper()
    days_back = st.sidebar.slider("Days of News", min_value=1, max_value=90, value=30)
    
    if ticker:
        try:
            # Display stock information
            display_stock_info(ticker)
            
            # Fetch news
            st.subheader(f"Recent News for {ticker}")
            news_df = get_stock_news(ticker, days_back)
            
            if news_df.empty:
                st.warning(f"No recent news found for {ticker}")
            else:
                # Display news articles
                for idx, article in news_df.iterrows():
                    with st.expander(f"{article['Date'].strftime('%Y-%m-%d')}: {article['Title']}"):
                        st.write(f"**Publisher:** {article['Publisher']}")
                        st.write(f"**Type:** {article['Type']}")
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
                                st.session_state.manual_urls[empty_slot] = article['Link']
                                st.success(f"Article added to URL slot {empty_slot + 1}!")
                            else:
                                # Add to research_urls if no empty slots
                                st.session_state.research_urls.add(article['Link'])
                                st.success("All URL slots full. Article added to additional research URLs!")
                
                # Download option
                csv = news_df.to_csv(index=False).encode('utf-8')
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
    
