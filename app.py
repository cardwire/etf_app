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

