import urllib.parse
import streamlit as st
from sqlalchemy import create_engine

# Define the connection string using the environment variables
credentials = {
    'server': st.secrets['AZURE_SQL_SERVER'],
    'database': st.secrets['AZURE_SQL_DATABASE'],
    'username': st.secrets['AZURE_SQL_USERNAME'],
    'password': st.secrets['AZURE_SQL_PASSWORD'],
    'driver': st.secrets['AZURE_SQL_DRIVER'],
    'port': st.secrets["AZURE_SQL_PORT"]
}

# Create the connection string
connection_string = f'mssql+pyodbc://{credentials["username"]}:{credentials["password"]}@{credentials["server"]}/{credentials["database"]}?driver={urllib.parse.quote_plus(credentials["driver"])}&TrustServerCertificate=yes'

# Create the engine
engine = create_engine(connection_string)



# Title
st.title("ETF-Finder")

def connect_to_server(): 
    try:
        with engine.connect() as connection:
            st.write("Connection successful!")
    except Exception as e:
        st.write(f"Connection failed: {e}")


st.subtitle('ETF Overview and Comparison Tool')
st.markdown(""" # /Disclaimer: Data provided by this app is not an official source. I can not guarantee for itsquality and actuality, 
    so take it witha grain of salt. Never use this app soleily to actually make financial decisiions with real money!""")

# Test the connection
st.text("Click on load database to connect with SQL-Server")
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
