import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
import pandas as pd


def get_umap_embeddings(data_final, n_components=3):
    reducer = umap.UMAP(n_components=n_components, metric='euclidean', n_neighbors=25, min_dist=0.5)
    embeddings = reducer.fit_transform(data_final)
    return pd.DataFrame(embeddings, columns=[f'UMAP{i+1}' for i in range(n_components)])

def get_principle_components(data_final, n_components=3):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_final)
    return pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

def get_t_sne(data_final, n_components=3):
    tsne = TSNE(n_components=n_components, metric='euclidean')
    tsne_components = tsne.fit_transform(data_final)
    return pd.DataFrame(tsne_components, columns=[f'TSNE{i+1}' for i in range(n_components)])


def get_nmf_components(data_final, n_components=3):
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    nmf_components = nmf.fit_transform(data_final)
    return pd.DataFrame(nmf_components, columns=[f'NMF{i+1}' for i in range(n_components)])



st.set_page_config(page_title="ETF UMAP", page_icon="📈")
st.markdown("# ETF Dimensionality Reduction")

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

# Dropdown for selecting dimensionality reduction method
dimensionality_reduction_method = st.selectbox("Select Dimensionality Reduction Method", options=["UMAP", "PCA", "t-SNE", "NMF"])

# Function to call the appropriate dimensionality reduction method
def call_dimensionality_reduction(method, data_final):
    if method == "UMAP":
        return get_umap_embeddings(data_final)
    elif method == "PCA":
        return get_principle_components(data_final)
    elif method == "t-SNE":
        return get_t_sne(data_final)
    else method == "NMF":
        return get_nmf_components(data_final, n_components=3)

# Button to launch 3D visualizer
if st.button("Launch 3D Visualizer"):
    data_embeddings = call_dimensionality_reduction(dimensionality_reduction_method, data_final)
    hover_data = data[['symbol', 'ytd_return', 'total_assets', 'fifty_day_average', 'bid', 'ask', 'category']]
    data_with_hover = pd.concat([data_embeddings, hover_data.reset_index(drop=True)], axis=1)
    
    # 3D Scatter plot
    fig = px.scatter_3d(data_with_hover, x=data_embeddings.columns[0], y=data_embeddings.columns[1], z=data_embeddings.columns[2], color=data['type'],
                        hover_data=hover_data.columns, title=f"3D {dimensionality_reduction_method} Clustering of ETFs")
    fig.update_traces(marker=dict(size=1.5), opacity=0.8)
    st.plotly_chart(fig)
