# 🔹 Açıklama: K-Means algoritması ile yapay veriler üzerinde kümeleme (clustering) yapar ve sonuçları görselleştirir.
# 🔹 Gerekli pip paketleri: pip install scikit-learn matplotlib

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Örnek veri oluşturma ---
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# --- Ham veriyi görselleştirme ---
plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], s=40, edgecolors="k", alpha=0.7)
plt.title("Örnek Veri Noktaları")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()

# --- K-Means modeli oluşturma ve eğitme ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans.fit(X)

# --- Küme etiketleri ve merkezleri ---
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# --- Sonuçların görselleştirilmesi ---
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40, edgecolors="k", alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Küme Merkezleri")
plt.title("K-Means Kümeleme Sonuçları")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()
