# 🔹 Açıklama: Farklı bağlantı (linkage) yöntemleriyle Agglomerative Clustering (Hiyerarşik Kümeleme) uygular ve dendrogram ile sonuçları görselleştirir.
# 🔹 Gerekli pip paketleri: pip install scikit-learn matplotlib scipy

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# --- Örnek veri oluşturma ---
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], s=40, edgecolors="k", alpha=0.7)
plt.title("Örnek Veri Noktaları")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()

# --- Kullanılacak bağlantı yöntemleri ---
linkage_methods = ["ward", "single", "average", "complete"]

# --- Şekil ayarları ---
plt.figure(figsize=(16, 8))

for i, linkage_method in enumerate(linkage_methods, 1):
    # 🔸 Model oluşturma
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    cluster_labels = model.fit_predict(X)
    
    # 🔸 Dendrogram oluşturma
    plt.subplot(2, 4, i)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    dendrogram(linkage(X, method=linkage_method), no_labels=True, color_threshold=0)
    plt.xlabel("Veri Noktaları")
    plt.ylabel("Uzaklık")

    # 🔸 Kümeleme görselleştirmesi
    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", s=40, edgecolors="k", alpha=0.7)
    plt.title(f"{linkage_method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")

plt.tight_layout()
plt.show()
