import urllib.parse
import streamlit as st
import pyodbc  # Ensure pyodbc is installed

# Load credentials from secrets
credentials = {
    'server': st.secrets['AZURE_SQL_SERVER'],
    'database': st.secrets['AZURE_SQL_DATABASE'],
    'username': st.secrets['AZURE_SQL_USERNAME'],
    'password': st.secrets['AZURE_SQL_PASSWORD'],
    'driver': st.secrets['AZURE_SQL_DRIVER'],
    'port': st.secrets["AZURE_SQL_PORT"]
}

# Initialize connection
@st.cache_resource
def init_connection():
    try:
        conn = pyodbc.connect(
            f"DRIVER={st.secrets['AZURE_SQL_DRIVER']};"
            f"SERVER={st.secrets['AZURE_SQL_SERVER']};"
            f"DATABASE={st.secrets['AZURE_SQL_DATABASE']};"
            f"UID={st.secrets['AZURE_SQL_USERNAME']};"
            f"PWD={st.secrets['AZURE_SQL_PASSWORD']};"
            "TrustServerCertificate=yes;"
        )
        st.write("✅ Connection successful!")
        return conn
    except Exception as e:
        st.error(f"🚨 Connection failed: {e}")
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

 etfs = get_etfs()
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
