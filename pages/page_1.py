import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="ETF Selector", page_icon=":chart_with_upwards_trend:")

st.markdown("# ETF Selection")

# Initialize session state variables
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []
    st.session_state.show_fund_data = False
    st.session_state.show_dividends = False

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes
edited_data = st.data_editor(data, column_config={
    "Select": st.column_config.CheckboxColumn("Select", help="Select up to 4 ETFs")
}, hide_index=True)

# Get selected ETFs
selected_etfs = edited_data[edited_data['Select']]['symbol'].tolist()

# Limit selection to 4 ETFs
if len(selected_etfs) > 4:
    st.warning("You can select up to 4 ETFs only.")
    selected_etfs = selected_etfs[:4]

# Update session state with the selected ETFs
st.session_state.selected_etfs = selected_etfs

# Display candlestick charts if ETFs are selected
if selected_etfs:
    st.markdown("## Candlestick Charts")
    cols = 2 if len(selected_etfs) > 1 else 1  # Determine layout
    rows = -(-len(selected_etfs) // cols)  # Ceiling division for rows
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_etfs)

    # Add the candlestick charts to the grid
    for i, etf in enumerate(selected_etfs):
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

st.divider()

# If ETF selection has been made, show additional sections for fund data and dividends
if selected_etfs:
    st.session_state.show_fund_data = True
    st.session_state.show_dividends = True

# Show Fund Data section if selected
if st.session_state.show_fund_data:
    # Get fund data for selected ETFs
    fund_data = []
    for etf in selected_etfs:
        ticker = yf.Ticker(etf)
        fund_data.append(ticker.info)  # Fetch fund info
    
    # Extract relevant fund data (example placeholders)
    sector_weightings = [data.get('sectorWeightings', {}) for data in fund_data]
    asset_classes = [data.get('assetClasses', {}) for data in fund_data]
    top_holdings = [data.get('topHoldings', {}) for data in fund_data]

    # Display sector weightings of all selected ETFs in one bar chart
    st.markdown("## Sector Weightings")
    fig = go.Figure()
    for i, sector_weighting in enumerate(sector_weightings):
        fig.add_trace(go.Bar(x=list(sector_weighting.keys()), y=list(sector_weighting.values()), name=selected_etfs[i]))

    fig.update_layout(barmode='group', title="Sector Weightings of Selected ETFs", xaxis_title="Sector", yaxis_title="Weighting")
    st.plotly_chart(fig)

    # Display asset classes of all selected ETFs in one bar chart
    st.markdown("## Asset Classes")
    fig = go.Figure()
    for i, asset_class in enumerate(asset_classes):
        fig.add_trace(go.Bar(x=list(asset_class.keys()), y=list(asset_class.values()), name=selected_etfs[i]))

    fig.update_layout(barmode='group', title="Asset Classes of Selected ETFs", xaxis_title="Asset Class", yaxis_title="Weight")
    st.plotly_chart(fig)

    # Display top holdings of all selected ETFs in one bar chart
    st.markdown("## Top Holdings")
    fig = go.Figure()
    for i, top_holding in enumerate(top_holdings):
        fig.add_trace(go.Bar(x=top_holding['Name'], y=top_holding['Holding Percent'], name=selected_etfs[i]))

    fig.update_layout(barmode='group', title="Top Holdings of Selected ETFs", xaxis_title="Holding", yaxis_title="Percent")
    st.plotly_chart(fig)

    st.divider()

# Show Dividends section if selected
if st.session_state.show_dividends:
    # Get dividends for selected ETFs
    dividends = [yf.Ticker(etf).dividends for etf in selected_etfs]

    # Display dividends of all selected ETFs in one line chart
    st.markdown("## Dividends")
    fig = go.Figure()
    for i, dividend in enumerate(dividends):
        fig.add_trace(go.Scatter(x=dividend.index, y=dividend, mode='lines', name=selected_etfs[i]))
    fig.update_layout(title="Dividends of Selected ETFs", xaxis_title="Date", yaxis_title="Dividend")

    st.plotly_chart(fig)

# Option to reset the selected ETFs and hide the sections
if st.button("Clear Selection"):
    st.session_state.selected_etfs = []
    st.session_state.show_fund_data = False
    st.session_state.show_dividends = False
    st.experimental_rerun()
