import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ETF Selector", page_icon=":chart_with_upwards_trend:")

st.markdown("# ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Add a column with checkboxes to allow selection
st.markdown("## Select up to 4 ETFs")
selected_etfs = []

for index, row in data.iterrows():
    if st.checkbox(row['symbol'], key=row['symbol']):
        selected_etfs.append(row['symbol'])
    
    # Limit selection to 4 ETFs
    if len(selected_etfs) > 4:
        st.warning("You can select up to 4 ETFs only.")
        break

# Display candlestick charts if ETFs are selected
if selected_etfs:
    st.markdown("## Candlestick Charts")
    cols = 2 if len(selected_etfs) > 1 else 1  # Determine layout
    rows = -(-len(selected_etfs) // cols)  # Ceiling division for rows
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_etfs)
    
    for i, etf in enumerate(selected_etfs):
        etf_data = pd.read_excel(f"database/{etf}.xlsx")  # Assuming each ETF has its own file
        candlestick = go.Candlestick(
            x=etf_data['Date'],
            open=etf_data['Open'],
            high=etf_data['High'],
            low=etf_data['Low'],
            close=etf_data['Close'],
            name=etf
        )
        fig.add_trace(candlestick, row=(i // cols) + 1, col=(i % cols) + 1)
    
    fig.update_layout(height=500 * rows, showlegend=False)
    st.plotly_chart(fig)
