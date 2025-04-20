import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import make_blobs

# Load model
with open('kmeans_model.pkl', 'rb') as f:
   loaded_model = pickle.load(f)

st.set_page_config(page_title="k-means Clustering ", layout="centered")

st.title("k-means Clustering Visualizer by Phitchayatida Pingkul")

# Generate example data
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict cluster labels using the loaded k-means model
y_kmeans = loaded_model.predict(X)

# Create scatter plot
fig, ax = plt.subplots()
scatter= ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
st.pyplot(fig)
