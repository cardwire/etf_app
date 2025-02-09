import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px

st.markdown("# ETF Finder")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

st.dataframe(data)

# change missing values to "other" in df.type
data['type'] = data['type'].fillna('Other')

# change "" to "Other" in df.type
data['type'] = data['type'].replace('-', 'Other')


# Count the occurrences of each type
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']

# Bar plot of df.type in plotly
fig = px.bar(type_counts, x='type', y='count', title='ETF Types')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='deepskyblue')
#remove "type" from x-axis
fig.update_xaxes(title_text='')
#center title
fig.update_layout(title_x=0.5)
fig.show()

st.plotly_chart(fig)

################################################################################################
st.divider()

# change missing values to "other" in df.category
data['category'] = data['category'].fillna('Other')

# change "" to "Other" in df.category
data['category'] = data['category'].replace('-', 'Other')


# Count the occurrences of each category
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

#Plot1: Bar plot of df.category in plotly

fig = px.bar(category_counts, x='count', y='category', title='ETF categories', orientation='h')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
# invert y-axis
fig.update_yaxes(categoryorder='total ascending')
#change figsize to double size of y-axis
fig.update_layout(height=1000)


#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors 
fig.update_traces(marker_color='pink')
#remove "category" from x-axis
fig.update_xaxes(title_text='')
#center title
fig.update_layout(title_x=0.5)
fig.show()

st.plotly_chart(fig)

################################################################################
st.divider()

# plot distribution of total_assets in plotly
x=np.log10(data['total_assets'])
fig = px.histogram(data, x=x, title='Distribution of Total Assets')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='yellow')
#remove "type" from x-axis
fig.update_xaxes(title_text='10exp(USD)')
#center title
fig.update_layout(title_x=0.5)
fig.show()

st.plotly_chart(fig)

##################################################################################
st.divider()

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

fig.show()
st.plotly_chart(fig)


# same for negative values with color red
x=np.log10(-data['ytd_return'][data['ytd_return']<0])
fig2 = px.histogram(data[data['ytd_return']<0], x=x, title='Distribution of YTD Return (Negative)')
#change background color to white, add gridlines in grey, and change font size
fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#set bar outlines to black
fig2.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig2.update_traces(marker_color='red')
#remove "type" from x-axis
fig2.update_xaxes(title_text='10exp(USD)')
#center title
fig2.update_layout(title_x=0.5)
#set y-axis to range from 0 to 350
fig2.update_yaxes(range=[0, 5])
#reduce binsize to 0.1
fig2.update_traces(histnorm='percent', xbins=dict(size=0.1))

fig2.show()
st.plotly_chart(fig2)
