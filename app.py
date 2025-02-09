Here is the content of your `app.py` file with some syntax errors that need fixing:

```python
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")

st.dataframe(etf_df)

st.divider()

def top10_dividend(etf_df):
    etf_df.sort_values("dividends", ascending=False).head(10)

st.button("Show Top 10 ETFs by Dividends", on_click=top10_dividend, args=(etf_df,))

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Function to toggle selection
def toggle_selection(symbol):
    if symbol in st.session_state.selected_etfs:
        st.session_state.selected_etfs.remove(symbol)
    else:
        st.session_state.selected_etfs.append(symbol)

# Display the dataframe with checkboxes
for i, row in etf_df.iterrows():
    checkbox = st.checkbox(
        label=f"{row['symbol']}",
        key=row['symbol'],
        value=row['symbol'] in st.session_state.selected_etfs
    )

    # Update session state based on the checkbox state
    if checkbox and row['symbol'] not in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])
    elif not checkbox and row['symbol'] in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])

# Display the selected ETFs
st.write("Selected ETFs:", st.session_state.selected_etfs)
```

Here are the changes made:
1. Corrected the `sort_values` method call.
2. Fixed the `on_click` parameter in `st.button`.
3. Adjusted the `toggle_selection` function to take a symbol instead of `etf_df.symbol`.
