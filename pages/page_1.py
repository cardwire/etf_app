import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

# Streamlit page config
st.set_page_config(page_title="ETF Selector", page_icon=":chart_with_upwards_trend:")

# Session state initialization
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

if 'fund_data' not in st.session_state:
    st.session_state.fund_data = []

if 'sector_weightings' not in st.session_state:
    st.session_state.sector_weightings = []

if 'asset_classes' not in st.session_state:
    st.session_state.asset_classes = []

if 'top_holdings' not in st.session_state:
    st.session_state.top_holdings = []

if 'dividends' not in st.session_state:
    st.session_state.dividends = []

st.markdown("# ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes for ETF selection
st.dataframe(data)

# Initialize a list for selected ETFs
selected_etfs = []

# Display checkboxes for each row in the dataframe
for index, row in data.iterrows():
    checkbox = st.checkbox(f"Select {row['fund_name']}", key=row['fund_name'], value=False)
    if checkbox:
        selected_etfs.append(row['symbol'])

# Limit selection to 4 ETFs
if len(selected_etfs) > 4:
    st.warning("You can select up to 4 ETFs only.")
    selected_etfs = selected_etfs[:4]

# Store selected ETFs in session state
st.session_state.selected_etfs = selected_etfs

# Clear All Data button
if st.button('Clear All Data'):
    st.session_state.selected_etfs = []
    st.session_state.fund_data = []
    st.session_state.sector_weightings = []
    st.session_state.asset_classes = []
    st.session_state.top_holdings = []
    st.session_state.dividends = []
    st.experimental_rerun()

# Fetch and store fund data in session state only if ETF is selected
if selected_etfs:
    fund_data = []
    sector_weightings = []
    asset_classes = []
    top_holdings = []
    dividends = []

    for etf in selected_etfs:
        ticker = yf.Ticker(etf)
        fund_data.append(ticker.history(period="1d", interval="1m"))
        
        # Corrected attributes for sector weightings, top holdings, asset classes, and dividends
        sector_weightings.append(ticker.info.get('sectorWeightings', {}))
        asset_classes.append(ticker.info.get('assetClass', {}))
        top_holdings.append(ticker.info.get('topHoldings', {}))
        dividends.append(ticker.dividends)

    st.session_state.fund_data = fund_data
    st.session_state.sector_weightings = sector_weightings
    st.session_state.asset_classes = asset_classes
    st.session_state.top_holdings = top_holdings
    st.session_state.dividends = dividends

# Display Candlestick charts if selected ETFs are available
if selected_etfs:
    st.markdown("## Candlestick Charts")
    cols = 2 if len(selected_etfs) > 1 else 1  # Determine layout
    rows = -(-len(selected_etfs) // cols)  # Ceiling division for rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_etfs)

    # Add the candlestick charts to the grid
    for i, etf in enumerate(selected_etfs):
        etf_data = st.session_state.fund_data[i]
        candlestick = go.Candlestick(
            x=etf_data.index,
            open=etf_data['Open'],
            high=etf_data['High'],
            low=etf_data['Low'],
            close=etf_data['Close'],
            name=etf
        )
        fig.add_trace(candlestick, row=(i // cols) + 1, col=(i % cols) + 1)

    fig.update_layout(height=500 * rows, showlegend=False)
    st.plotly_chart(fig)

# Display Sector Weightings if data is available
if st.session_state.sector_weightings:
    st.markdown("## Sector Weightings")
    fig = go.Figure()
    for i, sector_weighting in enumerate(st.session_state.sector_weightings):
        if sector_weighting:  # Check if data exists
            fig.add_trace(go.Bar(x=list(sector_weighting.keys()), y=list(sector_weighting.values()), name=selected_etfs[i]))

    fig.update_layout(barmode='group', title="Sector Weightings of Selected ETFs", xaxis_title="Sector", yaxis_title="Weighting")
    st.plotly_chart(fig)

# Display Asset Classes if data is available
if st.session_state.asset_classes:
    st.markdown("## Asset Classes")
    fig = go.Figure()
    for i, asset_class in enumerate(st.session_state.asset_classes):
        if asset_class:  # Check if data exists
            fig.add_trace(go.Bar(x=list(asset_class.keys()), y=list(asset_class.values()), name=selected_etfs[i]))

    fig.update_layout(barmode='group', title="Asset Classes of Selected ETFs", xaxis_title="Asset Class", yaxis_title="Weight")
    st.plotly_chart(fig)

# Display Top Holdings if data is available
if st.session_state.top_holdings:
    st.markdown("## Top Holdings")
    fig = go.Figure()
    for i, top_holding in enumerate(st.session_state.top_holdings):
        if isinstance(top_holding, pd.DataFrame) and 'symbol' in top_holding.columns:  # Ensure correct column exists
            fig.add_trace(go.Bar(x=top_holding['symbol'], y=top_holding['Holding Percent'], name=selected_etfs[i]))
        else:
            st.error(f"Top holdings for {selected_etfs[i]} do not have the expected 'symbol' column.")
    
    fig.update_layout(barmode='group', title="Top Holdings of Selected ETFs", xaxis_title="Holding", yaxis_title="Percent")
    st.plotly_chart(fig)

# Display Dividends if data is available
if st.session_state.dividends:
    st.markdown("## Dividends")
    fig = go.Figure()
    for i, dividend in enumerate(st.session_state.dividends):
        fig.add_trace(go.Scatter(x=dividend.index, y=dividend, mode='lines', name=selected_etfs[i]))
    
    fig.update_layout(title="Dividends of Selected ETFs", xaxis_title="Date", yaxis_title="Dividend")
    st.plotly_chart(fig)
