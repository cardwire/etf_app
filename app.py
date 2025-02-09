
import yfinance as yf
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

st.dataframe(etf_data)

st.divider()


