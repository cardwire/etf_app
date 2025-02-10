
import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import umap
from umap import UMAP
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Homepage", page_icon=":page_facing_up:")

st.sidebar.title("Navigation")
st.sidebar.success("Select a page below:")
page = st.sidebar.radio("Go to", ["Page 1", "Page 2", "Page 3"])

if page == "Homepage":
    st.markdown("# Homepage")
elif page == "Page 1":
    switch_page("page_1.py")
elif page == "Page 2":
    switch_page("page_2.py")
elif page == "Page 3":
    switch_page("page_3.py")

st.markdown("# ETF Finder")

# Load the ETF data
data = pd.read_excel("database/df.xlsx")
st.dataframe(data)



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
fig.update_traces(marker_color='green')
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

########################################################################
st.write("you selectdet SPY")

symbol = 'SPY' #implement selct function

ticker = yf.Ticker(f'{symbol}')
hist = ticker.history(period='1d', interval='1m')

# plot the price of SPY in plotly as candlestick chart
fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
fig.update_layout(title=f'{symbol} Todays Price', xaxis_title='Date', yaxis_title='Price')
#change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
#center title
fig.update_layout(title_x=0.5)

fig.show()

st.plotly_chart(fig)

###############################################################################
st.divider()

#drop currency
data = data.drop(columns = ["currency"], axis = 1)
# selcting only the numeric columns
data_numeric = data.select_dtypes(include=[np.number])

# Replace infinite values with NaN
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = IterativeImputer()
data_numeric_imputed = imputer.fit_transform(data_numeric)
data_numeric_imputed = pd.DataFrame(data_numeric_imputed, columns=data_numeric.columns)

# scale the numeric columns
scaler = StandardScaler()
data_numeric_imputed_scaled = scaler.fit_transform(data_numeric_imputed)
data_numeric_imputed_scaled = pd.DataFrame(data_numeric_imputed_scaled, columns=data_numeric_imputed.columns)

#selecting only the categorical columns
data_categorical = data.select_dtypes(include=[object])
#selecting only the categorical columns
data_categorical = data.select_dtypes(include=[object])
data_categorical.head()

cats_to_add = data_categorical[["type", "category"]]

# one-hot encode the categorical columns
cat_columns = pd.get_dummies(cats_to_add)
#transfer boolean to int
cat_columns = cat_columns.astype(int)

# combine the one-hot encoded categorical columns with the scaled numeric columns
data_final = pd.concat([data_numeric_imputed_scaled, cat_columns], axis=1)

#create a new column with the symbol of the ETF
data_final['symbol'] = data['symbol']

# calculate 3D umap clustering of the final dataframe
if 'symbol' in data_final.columns:
	data_final = data_final.drop(columns=["symbol"], axis=1)
  
reducer = umap.UMAP(n_components=3, metric='euclidean', n_neighbors=25, min_dist=0.5)
data_final_umap = reducer.fit_transform(data_final)
data_final_umap = pd.DataFrame(data_final_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])

# map categories to numeric values
category_mapping = {category: idx for idx, category in enumerate(data['category'].unique())}
data['category_numeric'] = data['type'].map(category_mapping)

# plot the 3D UMAP in plotly
hover_data = data[['symbol', 'ytd_return', 'total_assets', 'fifty_day_average', 'bid', 'ask', 'category']]
data_final_umap_with_hover = pd.concat([data_final_umap, hover_data.reset_index(drop=True)], axis=1)

fig = px.scatter_3d(data_final_umap_with_hover, x='UMAP1', y='UMAP2', z='UMAP3', color = data['type'], color_discrete_sequence=px.colors.qualitative.Dark2, hover_data=hover_data.columns)
fig.update_traces(marker=dict(size=1.5), opacity=0.8)
# change background color to white, add gridlines in grey, and change font size
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_size=12, yaxis=dict(gridcolor='lightgrey'))
# center title
fig.update_layout(title="3D representation of all ETFs by dimensionality reduction with umap",title_x=0.5)

fig.show()

st.plotly_chart(fig)

########################################
