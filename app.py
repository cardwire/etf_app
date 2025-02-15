import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="ETF Check", page_icon=":chart_with_upwards_trend:")
st.sidebar.success("Select a page below:")

page = st.sidebar.selectbox("Go to", ["ETF Check", "Latest Actions and current Performance", "ETF Inspector", "ETF 3D Visualizer", "ETF Forecast Tool"])

if page == "ETF App":
    st.markdown("## Disclaimer")
    st.divider()
    st.markdown("Modules")
    st.divider()
    st.markdown("Data provided")
    
    # Load ETF data
    data = pd.read_csv("etf_df.csv")
    st.dataframe(data)

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
    
elif page == "Latest Actions and current Performance":
    st.markdown("# ETF Statistics - Content Placeholder")
    st.write("This page provides an exploratory data analysis of our ETF-Database.")

elif page == "ETF 3D-Visualizer":
    st.markdown("# ETF 3D-Visualizer - Content Placeholder")
    st.write("This page provides 3D-representation of our ETF Database based on UMAP dimension reduction technique.")

elif page == "ETF Forecast Tool":
    st.markdown("# ETF Forecast Tool - Content Placeholder")
    st.write("This page provides a set of tools to get predictions concerning future performance of ETFs.")

st.divider()
# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Preprocess missing values
data['type'] = data['type'].fillna('Other').replace('-', 'Other')
data['category'] = data['category'].fillna('Other').replace('-', 'Other')

# Visualize data completeness
st.markdown("### Data Completeness at a glance! Here you see the distribution of missing values in the ETF-Database")
fig, ax = plt.subplots()
msno.matrix(data, ax=ax, color= (0.2549019607843137, 0.4117647058823529, 0.8823529411764706), fontsize = 8)
st.plotly_chart

# Count occurrences of each type and category
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Distribution of ETF types in the Database
st.markdown("### Distribution of ETF types in the Database")
fig_type = px.bar(type_counts, x='type', y='count', title='ETF Types')
fig_type.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_type.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_type.update_xaxes(title_text='')
fig_type.update_layout(title_x=0.5)
st.plotly_chart(fig_type)

st.divider()

# Distribution of ETF categories in the Database
st.markdown("### Distribution of ETF categories in the Database")
fig_category = px.bar(category_counts, x='count', y='category', title='ETF Categories', orientation='h')
fig_category.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_category.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'), height=1000)
fig_category.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig_category)

st.divider()

# Distribution of total assets
st.markdown("### Distribution of Total Assets")
x = np.log10(data['total_assets'])
fig_assets = px.histogram(data, x=x, title='Distribution of Total Assets')
fig_assets.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_assets.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig_assets.update_xaxes(title_text='10exp(USD)')
fig_assets.update_layout(title_x=0.5)
st.plotly_chart(fig_assets)


st.divider()

st.markdown("### Distribution of positive returns this year")
# plot distribution of ytd_return in plotly for positive values and negative values as separate plots in np.log10 scale
x=np.log10(data['ytd_return'][data['ytd_return']>0])
fig = px.histogram(data[data['ytd_return']>0], x=x, title='Distribution of YTD Return (Positive)')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='seagreen')
#remove "type" from x-axis
fig.update_xaxes(title_text='10exp(USD)')
#center title
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)


st.divider()

st.markdown("### Distribution of negative returns this year")
x=np.log10(-data['ytd_return'][data['ytd_return']<0])
fig = px.histogram(data[data['ytd_return']<0], x=x, title='Distribution of YTD Return (Negative)')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='red')
#remove "type" from x-axis
fig.update_xaxes(title_text='10exp(USD)')
#center title
fig.update_layout(title_x=0.5)
#set y-axis to range from 0 to 350
fig.update_yaxes(range=[0, 5])
#reduce binsize to 0.1
fig.update_traces(histnorm='percent', xbins=dict(size=0.1))
st.plotly_chart(fig)


st.divider()


st.markdown("### Distribution of returns for bids") 
x=np.log10(data['bid'])
fig = px.histogram(data, x=x, title='Distribution of Bid')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='orange')
#remove "type" from x-axis
fig.update_xaxes(title_text='10exp(USD)')
#center title
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)


st.divider()

st.markdown("### Distribution of returns for asks") 
x=np.log10(data['ask'])
fig = px.histogram(data, x=x, title='Distribution of Ask')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='orange')
#remove "type" from x-axis
fig.update_xaxes(title_text='10exp(USD)')
#center title
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)







