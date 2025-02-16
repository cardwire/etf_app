import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graphics.objects as go
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf



st.set_page_config(page_title="Latest Actions and current Performance", page_icon="📊")

st.markdown("# Select an ETF from the interactive Table. Or directly select your ETF of choice via searching its Tickersymbol")


# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Title
st.markdown("# ETF Selection")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Add a column for checkbox selection
data['Select'] = False

# Display dataframe with checkboxes
edited_data = st.data_editor(data, column_config={
    "Select": st.column_config.CheckboxColumn("Select", help="Select an ETF to get a forecast")
}, hide_index=True)

# Update selected ETFs in session state
selected_etfs = edited_data[edited_data['Select']]['symbol'].tolist()

# Limit selection to 1 ETFs
if len(selected_etfs) > 1:
    st.warning("You can select only one ETF to forecast its performance.")
    selected_etfs = selected_etfs[:1]

# Store selected ETFs in session state
st.session_state.selected_etfs = selected_etfs

# Check if an ETF is selected
if selected_etfs:
    # get symbol and ticker from selected ETF
    symbol = selected_etfs[0]
    ticker = yf.Ticker(symbol)

    # get long business summary
    long_sum = ticker.info['longBusinessSummary']

   

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Business Summary")
        st.markdown(f"### {symbol}:")
        st.markdown(f" {long_sum}")

    with col2:
        st.header("last 5 day performance")
        ticker_hist = ticker.history(period='5d', interval='1m')
        ticker_hist['Date'] = ticker_hist.index
        ticker_hist.reset_index(drop=True, inplace=True)
        ticker_hist['Date'] = pd.to_datetime(ticker_hist['Date'])
        ticker_hist['Date'] = ticker_hist['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Plotting the data as a candlestick chart using Plotly
        fig = go.Figure(data=[go.Candlestick(x=ticker_hist.index,
                                             open=ticker_hist['Open'],
                                             high=ticker_hist['High'],
                                             low=ticker_hist['Low'],
                                             close=ticker_hist['Close'])])

        fig.update_layout(title='5 day performance of GGG',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)
        
        # shift titel to center
        fig.update_layout(title_x=0.5)
        #reshape x-axis to range(1, 5 days)
        fig.update_xaxes(tickvals=np.arange(0, len(ggg_hist), step=78), ticktext=np.arange(0, 6))
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickprefix='$')
        fig.update_layout(showlegend=False)
        fig.update_layout(width=800, height=600)
        fig.update_layout(autosize=False)
        fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
        fig.update_layout(plot_bgcolor='white')
        fig.update_layout(paper_bgcolor='white')
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))
        fig.update_layout(yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'))       
        fig.update_layout(font=dict(family='Arial', size=12, color='black'))
        fig.update_layout(title_font=dict(size=20))
        fig.update_layout(title_font_size=20)
        fig.update_layout(title_font_family='Arial')
        fig.update_layout(title_font_color='black')
        
        st.plotly_chart(fig)

    with col3:
        st.header("")
        st.image("https://static.streamlit.io/examples/owl.jpg")
        st.markdown(f"## {symbol}:")
        st.markdown(f"### {long_sum}")

st.divider()
