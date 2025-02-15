import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno



st.set_page_config(page_title="ETF Visualizations", page_icon="📊")

st.markdown("# ETF Visualizations")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")

# Preprocess missing values
data['type'] = data['type'].fillna('Other').replace('-', 'Other')
data['category'] = data['category'].fillna('Other').replace('-', 'Other')

# visualize data completenes
st.write(msno.matrix(data))



# Count occurrences of each type and category
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Bar plot for ETF types
fig_type = px.bar(type_counts, x='type', y='count', title='ETF Types')
fig_type.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
st.plotly_chart(fig_type)

st.divider()

# Bar plot for ETF categories
fig_category = px.bar(category_counts, x='count', y='category', title='ETF Categories', orientation='h')
fig_category.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_category.update_yaxes(categoryorder='total ascending')
fig_category.update_layout(height=1000)
st.plotly_chart(fig_category)

st.divider()

# Plot distribution of total assets
fig_assets = px.histogram(data, x="total_assets", title="Distribution of Total Assets")
fig_assets.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
st.plotly_chart(fig_assets)

