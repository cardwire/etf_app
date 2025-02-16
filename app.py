import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from streamlit_extras.switch_page_button import switch_page

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Preprocess missing values
data['type'] = data['type'].fillna('Other').replace('-', 'Other')
data['category'] = data['category'].fillna('Other').replace('-', 'Other')

# Initialize Streamlit app
st.set_page_config(page_title="ETF Check", page_icon=":chart_with_upwards_trend:")
st.sidebar.success("Select a page below:")

# Page navigation
page = st.sidebar.selectbox("Go to", ["ETF Check", "Latest Actions and current Performance", "ETF Inspector", "ETF 3D Visualizer", "ETF Forecast Tool"])

if page == "ETF Check":
    st.markdown("# ETF Check")
    st.write("Welcome to the ETF Check app! Use the sidebar to navigate to different pages.")

elif page == "Latest Actions and current Performance":
    st.markdown("# Latest Actions and Current Performance")
    st.write("This page provides an exploratory data analysis of our ETF-Database.")

elif page == "ETF Inspector":
    st.markdown("# ETF Selection and Candlestick Chart")
    symbol = st.selectbox("Choose an ETF symbol:", data['symbol'].unique())
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='1d', interval='1m')
    
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                         open=hist['Open'],
                                         high=hist['High'],
                                         low=hist['Low'],
                                         close=hist['Close'])])
    fig.update_layout(title=f'{symbol} Price Chart', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig)

elif page == "ETF 3D Visualizer":
    st.markdown("# ETF 3D-Visualizer")
    st.write("This page provides 3D-representation of our ETF Database based on UMAP dimension reduction technique.")

elif page == "ETF Forecast Tool":
    st.markdown("# ETF Forecast Tool")
    st.write("This page provides a set of tools to get predictions concerning future performance of ETFs.")

# EDA of Entire Database Section
st.divider()
st.markdown("## Exploratory Data Analysis (EDA) of the ETF Database")

# Visualize data completeness
st.markdown("### Data Completeness at a Glance")
st.write("Here you see the distribution of missing values in the ETF-Database.")
fig, ax = plt.subplots()
msno.matrix(data, ax=ax, color=(0.2549019607843137, 0.4117647058823529, 0.8823529411764706), fontsize=8)
st.pyplot(fig)

# Count occurrences of each type and category
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Distribution of ETF types in the Database
st.markdown("### Distribution of ETF Types in the Database")
fig_type = px.bar(type_counts, x='type', y='count', title='ETF Types')
fig_type.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_type.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_type.update_xaxes(title_text='')
fig_type.update_layout(title_x=0.5)
st.plotly_chart(fig_type)

st.divider()

# Distribution of ETF categories in the Database
st.markdown("### Distribution of ETF Categories in the Database")
fig_category = px.bar(category_counts, x='count', y='category', title='ETF Categories', orientation='h')
fig_category.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_category.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'), height=1000)
fig_category.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig_category)

st.divider()

# Distribution of total assets
st.markdown("### Distribution of Total Assets")
x = np.log10(data['total_assets'][data['total_assets'] > 0])  # Filter out non-positive values
fig_assets = px.histogram(data, x=x, title='Distribution of Total Assets')
fig_assets.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_assets.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_assets.update_xaxes(title_text='10exp(USD)')
fig_assets.update_layout(title_x=0.5)
st.plotly_chart(fig_assets)

st.divider()

# Distribution of positive returns this year
st.markdown("### Distribution of Positive Returns This Year")
x = np.log10(data['ytd_return'][data['ytd_return'] > 0])  # Filter out non-positive values
fig = px.histogram(data[data['ytd_return'] > 0], x=x, title='Distribution of YTD Return (Positive)')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_traces(marker_line_color='black', marker_line_width=1, marker_color='seagreen')
fig.update_xaxes(title_text='10exp(USD)')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

st.divider()

# Distribution of negative returns this year
st.markdown("### Distribution of Negative Returns This Year")
x = np.log10(-data['ytd_return'][data['ytd_return'] < 0])  # Filter out non-negative values
fig = px.histogram(data[data['ytd_return'] < 0], x=x, title='Distribution of YTD Return (Negative)')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_traces(marker_line_color='black', marker_line_width=1, marker_color='seagreen')
fig.update_xaxes(title_text='10exp(USD)')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)
