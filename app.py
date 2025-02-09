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

st.set_page_config(page_title="Main Page", page_icon=":page_facing_up:")

st.sidebar.title("Navigation")
st.sidebar.success("Select a page below:")
page = st.sidebar.radio("Go to", ["Main Page", "Page 1", "Page 2", "Page 3"])

if page == "Main Page":
    st.markdown("# Main Page")
    st.write("Welcome to the Main Page")
elif page == "Page 1":
    switch_page("Page 1")
elif page == "Page 2":
    switch_page("Page 2")
elif page == "Page 3":
    switch_page("Page 3")



st.markdown("# ETF Finder")
