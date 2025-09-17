import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import time
import umap
from math import pi

np.random.seed(42)

data = pd.read_excel('data.xlsx', parse_dates=['timestamp'])  

features = ['PV_Active_Power', 'Temperature', 'Humidity', 'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation', 'Wind_Direction', 'Daily_Precipitation',
            'Tilted_Global_Radiation', 'Tilted_Diffuse_Radiation']
data_numeric = data[features]

if data_numeric.isnull().sum().any():
    data_numeric = data_numeric.fillna(data_numeric.mean())

data_numeric = data_numeric[(data_numeric['PV_Active_Power'] >= -5) &
                            (data_numeric['PV_Active_Power'] <= 400) &
                            (data_numeric['Global_Horizontal_Radiation'] <= 1400)]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

print("PCA...")
pca = PCA(n_components=0.90)  
data_pca = pca.fit_transform(data_scaled)
print(f"{data_pca.shape[1]}")

print("HDBSCAN...")
start_time = time.time()

clusterer = hdbscan.HDBSCAN(min_cluster_size=2500,  
                            min_samples=2,          
                            metric='euclidean',
                            cluster_selection_method='eom',
                            approx_min_span_tree=True)  

labels = clusterer.fit_predict(data_pca)

end_time = time.time()
print(f"{end_time - start_time:.2f}s")

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1
n_noise = list(labels).count(-1)
print(f"{n_clusters}")
print(f"{n_noise} ({n_noise / len(labels) * 100:.2f}%)")

data_clean = data.loc[data_numeric.index]  
data_clean['cluster'] = labels

other_features = features[1:]  
n_plots = len(other_features)
fig, axes = plt.subplots(nrows=(n_plots + 1) // 2, ncols=2, figsize=(15, 10))  
axes = axes.flatten()

for idx, feature in enumerate(other_features):
    sns.scatterplot(x=data_clean['PV_Active_Power'], y=data_clean[feature],
                    hue=data_clean['cluster'], palette='deep', size=1,
                    ax=axes[idx], legend='brief' if idx == 0 else False)
    axes[idx].set_title(f'Clustering: PV_Active_Power vs {feature}')
    axes[idx].set_xlabel('PV Active Power')
    axes[idx].set_ylabel(feature)

for idx in range(len(other_features), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()

plt.savefig('hdbscan_clustering_results.png', dpi=300, bbox_inches='tight')  
print("hdbscan_clustering_results.png")
plt.show()

cluster_stats = data_clean.groupby('cluster')[features].mean()
print(cluster_stats)

data_clean.to_excel('pv_data_clustered.xlsx', index=False)
print("pv_data_clustered.xlsx")

plt.rcParams["font.family"] = "Times New Roman"

data = pd.read_excel('pv_data_clustered.xlsx')

features = ['PV_Active_Power', 'Temperature', 'Humidity', 'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation', 'Wind_Direction', 'Daily_Precipitation',
            'Tilted_Global_Radiation', 'Tilted_Diffuse_Radiation']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
data_tsne = tsne.fit_transform(data_scaled)
tsne_df = pd.DataFrame(data_tsne, columns=["TSNE_1", "TSNE_2"])
tsne_df["cluster"] = data["cluster"]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tsne_df, x="TSNE_1", y="TSNE_2", hue="cluster", palette="deep", s=10, alpha=0.7)
plt.xlabel("t-SNE Component models_compare")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
sns.despine(top=True, right=True) 
plt.savefig("tsne_cluster_visualization.png", dpi=300)
plt.show()

reducer = umap.UMAP(n_components=2, random_state=42)
data_umap = reducer.fit_transform(data_scaled)
umap_df = pd.DataFrame(data_umap, columns=["UMAP_1", "UMAP_2"])
umap_df["cluster"] = data["cluster"]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=umap_df, x="UMAP_1", y="UMAP_2", hue="cluster", palette="deep", s=10, alpha=0.7)
plt.xlabel("UMAP Component models_compare")
plt.ylabel("UMAP Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
sns.despine(top=True, right=True)  
plt.savefig("umap_cluster_visualization.png", dpi=300)
plt.show()

cluster_stats = data.groupby('cluster')[features].agg(['mean', 'std'])
cluster_stats.to_excel("cluster_feature_stats.xlsx")
