
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")
etf_df = pd.read_csv("database/etf_df.csv")
etf_df.drop(columns = "maxAge", axis=1, inplace= True)



# Add a checkbox column
etf_df['select'] = False

st.dataframe(etf_df)


# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

def toggle_selection(symbol):
    if symbol in st.session_state.selected_etfs:
        st.session_state.selected_etfs.remove(symbol)
    else:
        st.session_state.selected_etfs.append(symbol)

# Display the scrollable table with checkboxes
for i, row in etf_df.iterrows():
    if len(st.session_state.selected_etfs) < 5 or row['symbol'] in st.session_state.selected_etfs:
        if st.checkbox(row['symbol'], key=row['symbol'], value=row['select']):
            toggle_selection(row['symbol'])

