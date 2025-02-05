
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







