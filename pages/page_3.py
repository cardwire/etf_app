import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ETF UMAP", page_icon="📈")

st.markdown("# ETF Dimensionality Reduction (UMAP)")

# Load ETF data
data = pd.read_excel("database/df.xlsx")

# Drop non-numeric columns and preprocess data
data = data.drop(columns=["currency"], errors='ignore')
data_numeric = data.select_dtypes(include=[np.number])
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
imputer = IterativeImputer()
data_imputed = imputer.fit_transform(data_numeric)
data_imputed = pd.DataFrame(data_imputed, columns=data_numeric.columns)

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)
data_scaled = pd.DataFrame(data_scaled, columns=data_numeric.columns)

# One-hot encode categorical variables
data_categorical = data.select_dtypes(include=[object])
cats_to_add = data_categorical[["type", "category"]]
cat_columns = pd.get_dummies(cats_to_add).astype(int)

# Combine processed numeric and categorical data
data_final = pd.concat([data_scaled, cat_columns], axis=1)

# Compute UMAP with 3D embedding
reducer = umap.UMAP(n_components=3, metric='euclidean', n_neighbors=25, min_dist=0.5)
data_umap = reducer.fit_transform(data_final)
data_umap = pd.DataFrame(data_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])

# Merge with ETF info for hover details
hover_data = data[['symbol', 'ytd_return', 'total_assets', 'fifty_day_average', 'bid', 'ask', 'category']]
data_umap_with_hover = pd.concat([data_umap, hover_data.reset_index(drop=True)], axis=1)

# 3D Scatter plot
fig_umap = px.scatter_3d(data_umap_with_hover, x='UMAP1', y='UMAP2', z='UMAP3', color=data['type'],
                         hover_data=hover_data.columns, title="3D UMAP Clustering of ETFs")
fig_umap.update_traces(marker=dict(size=1.5), opacity=0.8)
st.plotly_chart(fig_umap)
