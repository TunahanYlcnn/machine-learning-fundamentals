# 🔹 Açıklama: Farklı kümeleme algoritmalarının (KMeans, Spectral, DBSCAN, Birch vb.) çeşitli veri setleri üzerindeki sonuçlarını görselleştirir.
# 🔹 Gerekli pip paketleri: pip install scikit-learn numpy matplotlib

from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# --- Veri setleri oluştur ---
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = (np.random.rand(n_samples, 2), None)

# --- Kümeleme algoritmaları ve isimleri ---
clustering_names = [
    "MiniBatchKMeans",
    "SpectralClustering",
    "Ward",
    "Agglomerative(Average)",
    "DBSCAN",
    "Birch",
]

# --- Renk paleti ---
colors = np.array(["b", "g", "r", "c", "m", "y"])

# --- Veri seti listesi (modül ismini ezmemek için datasets_list kullandık) ---
datasets_list = [noisy_circles, noisy_moons, blobs, no_structure]

# --- Şekil oluştur ---
n_rows = len(datasets_list)
n_cols = len(clustering_names)
plt.figure(figsize=(3 * n_cols, 3 * n_rows))

i = 1
for i_dataset, dataset in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # --- Her veri seti için algoritmaları tanımla ---
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    spectral = cluster.SpectralClustering(n_clusters=2, assign_labels="discretize")
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    average_linkage = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    dbscan = cluster.DBSCAN(eps=0.2)
    birch = cluster.Birch(n_clusters=2)

    clustering_algorithms = [
        two_means,
        spectral,
        ward,
        average_linkage,
        dbscan,
        birch,
    ]

    # --- Her algoritmayı uygula ---
    for name, algo in zip(clustering_names, clustering_algorithms):
        try:
            algo.fit(X)
            if hasattr(algo, "labels_"):
                y_pred = algo.labels_.astype(int)
            else:
                y_pred = algo.predict(X)

            # Hatalı etiketleri (örnek: -1, outlier) güvenli çizim için modifiye et
            y_pred_safe = np.maximum(y_pred, 0) % len(colors)

            plt.subplot(n_rows, n_cols, i)
            if i_dataset == 0:
                plt.title(name, fontsize=10)
            plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred_safe], s=10)
            plt.xticks([])
            plt.yticks([])
        except Exception as e:
            plt.subplot(n_rows, n_cols, i)
            plt.text(0.5, 0.5, f"Error\n{name}", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])

        i += 1

plt.suptitle("Kümeleme Algoritmaları Karşılaştırması", fontsize=14)
plt.tight_layout()
plt.show()
