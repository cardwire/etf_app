
import yfinance as yf
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

st.dataframe(etf_df)

st.divider()

def get_tickers(etf_df):
    tickers = []
    for symbol in etf_df.symbol:
        tickers.append(yf.Ticker(symbol))
    return tickers

def top10_dividend(tickers):
    dividends = []
    for ticker in tickers:
        dividends.append(ticker.info.get('dividend', 0))
        
    return sorted(dividends, reverse=True)[:10]

tickers = get_tickers(etf_df)

st.button("Show Top 10 ETFs by Dividends", on_click=top10_dividend, args=(tickers,))

# Display the selected ETFs
st.write("Selected ETFs:", st.session_state.get('selected_etfs', []))
