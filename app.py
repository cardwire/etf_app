import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import umap
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="app", page_icon=":page_facing_up:")

st.sidebar.title("Navigation")
st.sidebar.success("Select a page below:")
page = st.sidebar.radio("Go to", ["Homepage", "Page 1", "Page 2", "Page 3"])

if page == "Homepage":
    st.markdown("# Homepage")
elif page == "Page 1":
    switch_page("page_1")
elif page == "Page 2":
    switch_page("page_2")
elif page == "Page 3":
    switch_page("page_3")

st.markdown("# ETF Finder")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")
st.dataframe(data)

# Change missing values to "Other" in df.type and df.category
data['type'] = data['type'].fillna('Other').replace('-', 'Other')
data['category'] = data['category'].fillna('Other').replace('-', 'Other')

# Count the occurrences of each type and category
type_counts = data['type'].value_counts().reset_index()
type_counts.columns = ['type', 'count']
category_counts = data['category'].value_counts().reset_index()
category_counts.columns = ['category', 'count']

# Bar plot of df.type in plotly
fig = px.bar(type_counts, x='type', y='count', title='ETF Types')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_traces(marker_line_color='black', marker_line_width=1, marker_color='deepskyblue')
fig.update_xaxes(title_text='')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

st.divider()

# Bar plot of df.category in plotly
fig = px.bar(category_counts, x='count', y='category', title='ETF Categories', orientation='h')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_yaxes(categoryorder='total ascending')
fig.update_layout(height=1000)
fig.update_traces(marker_line_color='black', marker_line_width=1, marker_color='pink')
fig.update_xaxes(title_text='')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

st.divider()

# Plot distribution of total_assets in plotly
x = np.log10(data['total_assets'])
fig = px.histogram(data, x=x, title='Distribution of Total Assets')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_traces(marker_line_color='black', marker_line_width=1, marker_color='yellow')
fig.update_xaxes(title_text='10exp(USD)')
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

st.divider()

# Plot distribution of ytd_return in plotly for positive and negative values
x_pos = np.log10(data['ytd_return'][data['ytd_return'] > 0])
fig_pos = px.histogram(data[data['ytd_return'] > 0], x=x_pos, title='Distribution of YTD Return (Positive)')
fig_pos.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_pos.update_traces(marker_line_color='black', marker_line_width=1, marker_color='green')
fig_pos.update_xaxes(title_text='10exp(USD)')
fig_pos.update_layout(title_x=0.5)
st.plotly_chart(fig_pos)

x_neg = np.log10(-data['ytd_return'][data['ytd_return'] < 0])
fig_neg = px.histogram(data[data['ytd_return'] < 0], x=x_neg, title='Distribution of YTD Return (Negative)')
fig_neg.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig_neg.update_traces(marker_line_color='black', marker_line_width=1, marker_color='red')
fig_neg.update_xaxes(title_text='10exp(USD)')
fig_neg.update_layout(title_x=0.5)
fig_neg.update_yaxes(range=[0, 5])
fig_neg.update_traces(histnorm='percent', xbins=dict(size=0.1))
st.plotly_chart(fig_neg)

st.write("You selected SPY")

symbol = 'SPY'  # Implement select function
ticker = yf.Ticker(f'{symbol}')
hist = ticker.history(period='1d', interval='1m')

# Plot the price of SPY in plotly as candlestick chart
fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                     open=hist['Open'],
                                     high=hist['High'],
                                     low=hist['Low'],
                                     close=hist['Close'])])
fig.update_layout(title=f'{symbol} Todays Price', xaxis_title='Date', yaxis_title='Price')
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_layout(title_x=0.5)
st.plotly_chart(fig)

st.divider()

# Drop currency column and select only numeric columns
data = data.drop(columns=["currency"], axis=1)
data_numeric = data.select_dtypes(include=[np.number])

# Replace infinite values with NaN and impute missing values
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = IterativeImputer()
data_numeric_imputed = imputer.fit_transform(data_numeric)
data_numeric_imputed = pd.DataFrame(data_numeric_imputed, columns=data_numeric.columns)

# Scale the numeric columns
scaler = StandardScaler()
data_numeric_imputed_scaled = scaler.fit_transform(data_numeric_imputed)
data_numeric_imputed_scaled = pd.DataFrame(data_numeric_imputed_scaled, columns=data_numeric_imputed.columns)

# Select and one-hot encode categorical columns
data_categorical = data.select_dtypes(include=[object])
cats_to_add = data_categorical[["type", "category"]]
cat_columns = pd.get_dummies(cats_to_add).astype(int)

# Combine the one-hot encoded categorical columns with the scaled numeric columns
data_final = pd.concat([data_numeric_imputed_scaled, cat_columns], axis=1)
data_final['symbol'] = data['symbol']

# Calculate 3D UMAP clustering of the final dataframe
if 'symbol' in data_final.columns:
    data_final = data_final.drop(columns=["symbol"], axis=1)

reducer = umap.UMAP(n_components=3, metric='euclidean', n_neighbors=25, min_dist=0.5)
data_final_umap = reducer.fit_transform(data_final)
data_final_umap = pd.DataFrame(data_final_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])

# Map categories to numeric values
category_mapping = {category: idx for idx, category in enumerate(data['category'].unique())}
data['category_numeric'] = data['type'].map(category_mapping)

# Plot the 3D UMAP in plotly
hover_data = data[['symbol', 'ytd_return', 'total_assets', 'fifty_day_average', 'bid', 'ask', 'category']]
data_final_umap_with_hover = pd.concat([data_final_umap, hover_data.reset_index(drop=True)], axis=1)

fig = px.scatter_3d(data_final_umap_with_hover, x='UMAP1', y='UMAP2', z='UMAP3', color=data['type'],
                    color_discrete_sequence=px.colors.qualitative.Dark2, hover_data=hover_data.columns)
fig.update_traces(marker=dict(size=1.5), opacity=0.8)
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
fig.update_layout(title="3D representation of all ETFs by dimensionality reduction with umap", title_x=0.5)
st.plotly_chart(fig)
