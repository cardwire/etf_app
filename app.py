


import urllib.parse
import streamlit as st
import sqlalchemy
from sqlalchemy import create_engine
import pyodbc  # Ensure pyodbc is installed
import pypyodbc as odbc_driver
# Load credentials from secrets
credentials = {
    'server': st.secrets['AZURE_SQL_SERVER'],
    'database': st.secrets['AZURE_SQL_DATABASE'],
    'username': st.secrets['AZURE_SQL_USERNAME'],
    'password': st.secrets['AZURE_SQL_PASSWORD'],
    'driver': st.secrets['AZURE_SQL_DRIVER'],
    'port': st.secrets["AZURE_SQL_PORT"]
}

# Adjust ODBC connection string
connection_string = f"DRIVER={{{credentials['driver']}}};SERVER={credentials['server']},{credentials['port']};DATABASE={credentials['database']};UID={credentials['username']};PWD={credentials['password']};TrustServerCertificate=yes"

# Create PyODBC connection
def connect_to_server():
    try:
        conn = pyodbc.connect(connection_string)
        st.write("✅ Connection successful!")
        conn.close()
    except Exception as e:
        st.write(f"❌ Connection failed: {e}")

st.title("ETF-Finder")
st.button(label="Load Database", type="primary", on_click=connect_to_server)



# ETF Filtering
# etfs = get_etfs()
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
