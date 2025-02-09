import yfinance as yf
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

st.dataframe(etf_df)

st.divider()

def get_tickers(etf_df):
    for symbol in etf_df.symbol:
        ticker = []
        ticker = ticker.append(yf.Ticker("symbol"))

def top10_dividend(ticker):
    for ticker in ticker:
        dividends = ticker.info().dividend()

     return dividends.srt_values("ascending" = False).head(10)

st.button("Show Top 10 ETFs by Dividends", on_click=top10_dividend, args=(ticker,))





# Display the selected ETFs
st.write("Selected ETFs:", st.session_state.selected_etfs)

