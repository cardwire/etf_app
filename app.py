To limit the selection to a maximum of 5 ETFs via the multiselect, you need to modify the multiselect component in the `app.py` file. Here is the updated code:

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
import plotly.offline as pyo
import yfinance as yf

st.markdown("# ETF Finder")
etf_df = pd.read_csv("database/etf_df.csv")

st.text("select up to 5 ETFs for multiple comparisons")
selected_etfs = st.multiselect("Select ETFs", etf_df['symbol'], max_selections=5)
st.write("You selected:", selected_etfs)

# Display ETF Details
if selected_etfs:
    for etf in selected_etfs:
        st.header(etf)
        # Assuming get_etf_data is a function to get ETF data
        # etf_data = get_etf_data(etf)
        # st.subheader('Info')
        # st.write(etf_data['info'])
        # st.subheader('Holdings')
        # st.write(etf.holdings)
        # st.subheader('Dividends')
        # st.write(etf_data['dividends'])
        # st.subheader('History')
        # st.write(etf_data['history'])
        # st.subheader('Candlestick Plot')
        # st.line_chart(etf_data['history']['Close'])
        # st.subheader('Sustainability')
        # st.write(etf_data['sustainability'])
```

Replace the existing multiselect component with the above code snippet to apply the selection limit.
