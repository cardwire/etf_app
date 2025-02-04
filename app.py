import streamlit as st
import psychopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
from os import getenv
import urllib.parse

# Define the connection string using the environment variables
credentials = {
server : st.secrets['AZURE_SQL_SERVER'],
database : st.secrets['AZURE_SQL_DATABASE'],
username : st.secrets['AZURE_SQL_USERNAME'],
password :  st.secrets['AZURE_SQL_PASSWORD'],
driver : st.secrets['AZURE_SQL_DRIVER'],
port : st.secrets["AZURE_SQL-PORT"]
}

#TITEL
st.titel("ETF-Finder")


# Create the connection string
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={urllib.parse.quote_plus(driver)}&TrustServerCertificate=yes'

# Create the engine
engine = create_engine(connection_string)

def connect_to_server(): 
    try:
        with engine.connect() as connection:
            print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")




# Test the connection
st.text("click on load database to connect with SQL-Server")

st.button(label= "Load Database", type = "primary", on_click = connect_to_server())
    




st.title('ETF Overview and Comparison Tool')

# ETF Filtering
#etfs = get_etfs()
#selected_etfs = st.multiselect('Select up to 5 ETFs to compare', etfs, default=etfs[:5])



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
