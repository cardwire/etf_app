import streamlit as st
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# UMAP
def get_umap_embeddings(data_final, n_components=3):
    reducer = umap.UMAP(n_components=n_components, metric='euclidean', n_neighbors=25, min_dist=0.5)
    embeddings = reducer.fit_transform(data_final)
    return pd.DataFrame(embeddings, columns=[f'UMAP{i+1}' for i in range(n_components)])

# PCA
def get_principle_components(data_final, n_components=3):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_final)
    return pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

# t-SNE
def get_t_sne(data_final, n_components=3):
    tsne = TSNE(n_components=n_components, metric='euclidean')
    tsne_components = tsne.fit_transform(data_final)
    return pd.DataFrame(tsne_components, columns=[f'TSNE{i+1}' for i in range(n_components)])

# NMF
def get_nmf_components(data_final, n_components=3):
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    nmf_components = nmf.fit_transform(data_final)
    return pd.DataFrame(nmf_components, columns=[f'NMF{i+1}' for i in range(n_components)])

# LDA
def get_lda_components(data_final, labels, n_components=3):
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(data_final, labels)
    return pd.DataFrame(lda_components, columns=[f'LDA{i+1}' for i in range(n_components)])

# ICA
def get_ica_components(data_final, n_components=3):
    ica = FastICA(n_components=n_components, random_state=0)
    ica_components = ica.fit_transform(data_final)
    return pd.DataFrame(ica_components, columns=[f'ICA{i+1}' for i in range(n_components)])

# Generalized Discriminant Analysis (GDA)
def get_gda_components(data_final, labels, n_components=3):
    # GDA is not directly available in scikit-learn, so we use KernelPCA as an approximation
    from sklearn.decomposition import KernelPCA
    gda = KernelPCA(n_components=n_components, kernel='rbf')
    gda_components = gda.fit_transform(data_final)
    return pd.DataFrame(gda_components, columns=[f'GDA{i+1}' for i in range(n_components)])

# Missing Values Ratio (MVR)
def apply_mvr(data_final, threshold=0.5):
    missing_values_ratio = data_final.isnull().mean()
    return data_final.loc[:, missing_values_ratio < threshold]

# Main function to call dimensionality reduction methods
def call_dimensionality_reduction(method, data_final, labels=None, n_components=3):
    if method == "UMAP":
        return get_umap_embeddings(data_final, n_components)
    elif method == "PCA":
        return get_principle_components(data_final, n_components)
    elif method == "t-SNE":
        return get_t_sne(data_final, n_components)
    elif method == "NMF":
        return get_nmf_components(data_final, n_components)
    elif method == "LDA":
        return get_lda_components(data_final, labels, n_components)
    elif method == "ICA":
        return get_ica_components(data_final, n_components)
    elif method == "GDA":
        return get_gda_components(data_final, labels, n_components)
    elif method == "MVR":
        return apply_mvr(data_final)


# Streamlit App
st.set_page_config(page_title="ETF Dimensionality Reduction", page_icon="📈")
st.markdown("# ETF Dimensionality Reduction")

# Load ETF data
data = pd.read_excel("database/df.xlsx")
data["type"] = data['type'].fillna("Not Provided")
data["type"] = data['type'].replace("-", "Typeless")
labels = data["type"]

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
cats_to_add = data_categorical[["type", "category"]].copy()
cats_to_add['type'] = cats_to_add['type'].fillna('Unknown')
cat_columns = pd.get_dummies(cats_to_add).astype(int)

# Combine processed numeric and categorical data
data_final = pd.concat([data_scaled, cat_columns.reset_index(drop=True)], axis=1)

# Dropdown for selecting dimensionality reduction method
methods = [
    "UMAP", 
    "PCA", 
    "t-SNE", 
    "NMF", 
    "LDA", 
    "ICA", 
    "GDA", 
    "MVR"
]

dimensionality_reduction_method = st.selectbox("Select Dimensionality Reduction Method", options=methods)

# Button to launch 3D visualizer
if st.button("Launch 3D Visualizer"):
    data_embeddings = call_dimensionality_reduction(dimensionality_reduction_method, data_final, labels)
    hover_data = data[['symbol', 'ytd_return', 'total_assets', 'fifty_day_average', 'bid', 'ask', 'category']].copy()
    data_with_hover = pd.concat([data_embeddings.reset_index(drop=True), hover_data.reset_index(drop=True)], axis=1)
    
    # 3D Scatter plot
    if data_embeddings.shape[1] >= 3:
        fig = px.scatter_3d(data_with_hover, x=data_embeddings.columns[0], y=data_embeddings.columns[1], z=data_embeddings.columns[2], color=labels,
                            hover_data=hover_data.columns, title=f"3D {dimensionality_reduction_method} Clustering of ETFs")
        fig.update_traces(marker=dict(size=2.5), opacity=0.8)
        st.plotly_chart(fig)
    else:
        st.error("Selected method does not produce enough components for 3D visualization.")
