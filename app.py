To implement a scrollable database where clicking once on a symbol selects it and clicking again deselects it, along with a confirm button, follow these steps:

1. Load the ETF data into a DataFrame.
2. Create a scrollable table for selecting/deselecting ETFs.
3. Implement logic to handle selection/deselection.
4. Add a confirm button to finalize the selection.

Here is the updated code:

```python
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")
etf_df = pd.read_csv("database/etf_df.csv")

# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

def toggle_selection(symbol):
    if symbol in st.session_state.selected_etfs:
        st.session_state.selected_etfs.remove(symbol)
    else:
        st.session_state.selected_etfs.append(symbol)

st.text("Click once on a symbol to select, click again to deselect. Max 5 selections.")

# Display the scrollable table
for i, row in etf_df.iterrows():
    if len(st.session_state.selected_etfs) < 5 or row['symbol'] in st.session_state.selected_etfs:
        if st.button(row['symbol'], key=row['symbol']):
            toggle_selection(row['symbol'])

st.write("Selected ETFs:", st.session_state.selected_etfs)

# Confirm button
if st.button("Confirm Selection"):
    confirmed_etfs = st.session_state.selected_etfs
    st.write("You confirmed:", confirmed_etfs)
    # Add logic to handle confirmed ETFs (e.g., display details)
```

Replace the existing code with this snippet to implement the new selection method.
