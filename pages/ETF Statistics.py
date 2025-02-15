import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno
import numpy as np



st.set_page_config(page_title="ETF Visualizations", page_icon="📊")

st.markdown("# ETF Visualizations")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Preprocess missing values
data['type'] = data['type'].fillna('Other').replace('-', 'Other')
data['category'] = data['category'].fillna('Other').replace('-', 'Other')

import matplotlib.pyplot as plt

# visualize data completeness
st.markdown(" ### Data Completeness at a glance! here you see the distribution of missing values in the ETF-Database") 
fig, ax = plt.subplots()
msno.matrix(data, ax=ax)
st.pyplot(fig)



# Count occurrences of each type and category
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']


################################################################################

st.markdown("### Distribution of ETF types in the Database")
# Bar plot for ETF types
fig_type = px.bar(type_counts, x='type', y='count', title='ETF Types')
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

st.plotly_chart(fig_type)

#####################################################################################################

st.divider()

#######################################################################################################

st.markdown("### Distribution of ETF categories in the Database")

# Bar plot for ETF categories
fig_category = px.bar(category_counts, x='count', y='category', title='ETF Categories', orientation='h')
fig.update_traces(marker_line_color='black', marker_line_width=1)
#set bar colors to deepskyblueblue
fig.update_traces(marker_color='deepskyblue')
fig_category.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_category.update_yaxes(categoryorder='total ascending')
fig_category.update_layout(height=1000)
st.plotly_chart(fig_category)


#########################################################################################################

st.divider()

############################################################################################################

st.markdown("### Distribution of ETF categories in the Database")

# Plot distribution of total assets
x=np.log10(data['total_assets'])
fig = px.histogram(df, x=x, title='Distribution of Total Assets')
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
