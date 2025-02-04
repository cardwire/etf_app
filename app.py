import streamlit as st
from database.operations import get_etfs, get_etf_details
from yfinance_integration.yfinance_api import get_etf_data

st.title('ETF Overview and Comparison Tool')

# ETF Filtering
etfs = get_etfs()
selected_etfs = st.multiselect('Select up to 5 ETFs to compare', etfs, default=etfs[:5])



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
