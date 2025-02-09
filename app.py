
import streamlit as st
import pandas as pd

st.markdown("# ETF Finder")

# Load the ETF data
etf_df = pd.read_csv("database/etf_df.csv")


st.dataframe(etf_df)


st.divider()



'''
# Initialize session state for selected ETFs
if 'selected_etfs' not in st.session_state:
    st.session_state.selected_etfs = []

# Function to toggle selection
def toggle_selection(etf_df.symbol):
   
    if etf_df.symbol in st.session_state.selected_etfs:
        st.session_state.selected_etfs.remove(etf_df.symbol)
    else:
        st.session_state.selected_etfs.append(etf_df.symbol)

# Display the dataframe with checkboxes
for i, row in etf_df.iterrows():  # Fix: Correct way to iterate over DataFrame rows
    checkbox = st.checkbox(
        label=f"{row['symbol']}",  # Fix: Added label to display ETF symbol
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

'''
