import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import yfinance as yf
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="ETF Selector", page_icon=":chart_with_upwards_trend:")

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Title
st.markdown("# ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")
data = data[["symbol", "full_name", "type", "total_assets", "ytd_return"]]
data = data.rename(columns={"full_name": "funds name", "type": "funds type", "total_assets": "AUM"})


# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes
edited_data = st.data_editor(data, column_config={
    "Select": st.column_config.CheckboxColumn("Select", help="Select up to 4 ETFs")
}, hide_index=True)

# Update selected ETFs in session state
selected_etfs = edited_data[edited_data['Select']]['symbol'].tolist()

# Limit selection to 4 ETFs
if len(selected_etfs) > 4:
    st.warning("You can select up to 4 ETFs only.")
    selected_etfs = selected_etfs[:4]

# Store selected ETFs in session state
st.session_state.selected_etfs = selected_etfs

# Display candlestick charts if ETFs are selected
if st.session_state.selected_etfs:
    st.markdown("## Candlestick Charts")
    cols = 2 if len(st.session_state.selected_etfs) > 1 else 1  # Determine layout
    rows = -(-len(st.session_state.selected_etfs) // cols)  # Ceiling division for rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=st.session_state.selected_etfs)

    # Add the candlestick charts to the grid
    for i, etf in enumerate(st.session_state.selected_etfs):
        ticker = yf.Ticker(etf)
        etf_data = ticker.history(period="1d", interval="1m")
        
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

# Show Funds Data and Dividends sections only if ETFs are selected
if st.session_state.selected_etfs:
    st.divider()

    # Get fund data for selected ETFs
    fund_data = []
    for etf in st.session_state.selected_etfs:
        ticker = yf.Ticker(etf)
        fund_data.append(ticker.get_funds_data())

    # Get sector weightings for selected ETFs
    sector_weightings = [data.sector_weightings for data in fund_data]

    # Get asset classes for selected ETFs
    asset_classes = [data.asset_classes for data in fund_data]

    # Get top holdings for selected ETFs
    top_holdings = [data.top_holdings for data in fund_data]

    # Display sector weightings of all selected ETFs in one bar chart
    st.markdown("## Sector Weightings")
    fig = go.Figure()
    for i, sector_weighting in enumerate(sector_weightings):
        fig.add_trace(go.Bar(x=list(sector_weighting.keys()), y=list(sector_weighting.values()), name=st.session_state.selected_etfs[i]))

    fig.update_layout(barmode='group', title="Sector Weightings of Selected ETFs", xaxis_title="Sector", yaxis_title="Weighting",)
    fig.update_traces(marker_line_color='black',  marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig)

    # Display asset classes of all selected ETFs in one bar chart
    st.markdown("## Asset Classes")
    fig = go.Figure()
    for i, asset_class in enumerate(asset_classes):
        fig.add_trace(go.Bar(x=list(asset_class.keys()), y=list(asset_class.values()), name=st.session_state.selected_etfs[i]))
        
    fig.update_traces(marker_line_color='black',  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(barmode='group', title="Asset Classes of Selected ETFs", xaxis_title="Asset Class", yaxis_title="Weight", )
    st.plotly_chart(fig)

    # Display top holdings of all selected ETFs in one bar chart
    st.markdown("## Top Holdings")
    fig = go.Figure()
    for i, top_holding in enumerate(top_holdings):
        fig.add_trace(go.Bar(x=top_holding['Name'], y=top_holding['Holding Percent'], name=st.session_state.selected_etfs[i]))
    
    fig.update_traces(marker_line_color='black',  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(barmode='group', title="Top Holdings of Selected ETFs", xaxis_title="Holding", yaxis_title="Percent",)
    st.plotly_chart(fig)

    st.divider()

    # Get dividends for selected ETFs
    dividends = [yf.Ticker(etf).dividends for etf in st.session_state.selected_etfs]

    # Display dividends of all selected ETFs in one line chart
    st.markdown("## Dividends")
    fig = go.Figure()
    for i, dividend in enumerate(dividends):
        fig.add_trace(go.Scatter(x=dividend.index, y=dividend, mode='lines', name=st.session_state.selected_etfs[i]))
    fig.update_layout(title="Dividends of Selected ETFs", xaxis_title="Date", yaxis_title="Dividend")
    
    st.plotly_chart(fig)

# Add a button to clear selections
if st.button("Clear Selections"):
    st.session_state.selected_etfs = []
    st.experimental_rerun()
