import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.offline as pyo
import yfinance as yf



st.markdown("# ETF Finder")
etf_df = pd.read_csv("database/etf_df.csv")


st.text("select up to 5 ETFs for multiple comparisons")
options = "select up to 5 ETFs", [etf_df.symbol] 
st.multiselect("select ETFs",options = f"{etf_df.symbol}")
#st.button("Load Database", type= "primary", on_click = st.dataframe(etf_df) )
st.write("you selected", options,  max_selections=5)
# ETF Filtering
# etfs = get_etfs()
# selected_etfs = st.multiselect('Select up to 5 ETFs to compare', etfs, default=etfs[:5])
# ETF Filtering
# etfs = get_etfs()
# selected_etfs = st.multiselect('Select up to 5 ETFs to compare', etfs, default=etfs[:5])

 #etfs = get_etfs()
# selected_etfs = st.multiselect('Select up to 5 ETFs to compare', etfs, default=etfs[:5])

'''
# Display ETF Details
for etf in selected_etfs:
    st.header(etf.name)
    etf_data = get_etf_data(etf.symbol)
    st.subheader('Info')
    st.write(etf_data['info'])
    st.subheader('Holdings')
    st.write(etf.holdings)
    st.subheader('Dividends')
    st.write(etf_data['dividends'])
    st.subheader('History')
    st.write(etf_data['history'])
    st.subheader('Candlestick Plot')
    st.line_chart(etf_data['history']['Close'])
    st.subheader('Sustainability')
    st.write(etf_data['sustainability'])
'''
