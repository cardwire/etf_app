
import streamlit as st
import pandas as pd
import yfinance as yf



st.markdown("# ETF Finder")

etf_df = pd.read_csv("database/etf_df.csv")


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

# Update the selection based on the 'select' column
for i, row in etf_df.iterrows():
    if row['select'] and row['symbol'] not in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])
    elif not row['select'] and row['symbol'] in st.session_state.selected_etfs:
        toggle_selection(row['symbol'])

st.write("Selected ETFs:", st.session_state.selected_etfs)

# JavaScript for handling checkbox click event
st.markdown("""
    <script>
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('click', function() {
            if (this.checked) {
                this.nextSibling.textContent = 'x';
            } else {
                this.nextSibling.textContent = '';
            }
        });
    });
    </script>
""", unsafe_allow_html=True)
