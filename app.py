import streamlit as st
import pyodbc as ODBC
import sqlalchemy
from sqlalchemy import create_engine


# Load credentials from secrets
credentials = {
    'server': st.secrets['AZURE_SQL_SERVER'],
    'database': st.secrets['AZURE_SQL_DATABASE'],
    'username': st.secrets['AZURE_SQL_USERNAME'],
    'password': st.secrets['AZURE_SQL_PASSWORD'],
   
   
}

# Initialize connection
@st.cache_resource
def init_connection():
    try:
        conn_str = (f"Server=tcp:myfreesqldbserverjonny.database.windows.net,1433;"+
            "Initial Catalog=myFreeDB;"+
            "driver=ODBC Driver 17 for SQL Server"+
            "Persist Security Info=False;"+
            "User ID={username};"+
            "MultipleActiveResultSets=False;"+
            "Encrypt=True;"+
            "TrustServerCertificate=False;"+
            "Authentication=Active Directory Integrated"
            )
            
        conn = pyodbc.connect(conn_str)
        st.write("✅ Connection successful!")
        return conn
    except Exception as e:
        st.error(f"🚨 Connection failed: {e}")
        st.error(f"Connection string used: {conn_str}")
        return None

conn = init_connection()

# Perform query
@st.cache_data(ttl=600)
def run_query(query):
    if conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    return []

# Run SQL Query
rows = run_query("SELECT * FROM mytable;")

# Print results
if rows:
    for row in rows:
        st.write(f"{row[0]} has a :{row[1]}:")




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
