
import yfinance as yf
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

st.dataframe(df)



