import streamlit as st
import pandas as pd
import plotly.express as px
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Latest Actions and current Performance", page_icon="📊")

st.markdown("# Select an ETF from the interactive Table. Or directly select your ETF of choice via searching its Tickersymbol")


data = pd.read_csv("df_esg.csv", index_col=False)

data["select"] = st.checkbox()
st.dataframe(data)
st.write()
st.text_input()
