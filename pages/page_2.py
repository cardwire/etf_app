import streamlit as st

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
