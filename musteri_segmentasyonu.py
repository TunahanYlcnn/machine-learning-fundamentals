# 🔹 Açıklama: Müşteri verilerini KMeans ve Hiyerarşik Kümeleme (Dendrogram) yöntemleriyle segmente eder ve sonuçları görselleştirir.
# 🔹 Gerekli pip paketleri: pip install pandas matplotlib numpy scikit-learn scipy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# --- Veri setini yükleme ---
# Not: Pickle dosyası, önceden hazırlanmış müşteri verilerini içerir.
data = pd.read_pickle("/content/4_5_GozetimsizOgrenmeDurumCalismasi")
X = data.values

# --- Veriyi pozitif hale getirme (görselleştirme için) ---
X[:, 0] = np.abs(2 * np.min(X[:, 0])) + X[:, 0]
X[:, 1] = np.abs(2 * np.min(X[:, 1])) + X[:, 1]

# --- Ham veri görselleştirme ---
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.7, edgecolors="k")
plt.title("Müşteri Verisi")
plt.xlabel("Gelir (Income)")
plt.ylabel("Harcama Skoru (Spending Score)")

# --- KMeans kümeleme ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
kmeans.fit(X)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# --- Sonuçların görselleştirilmesi ---
plt.figure(figsize=(14, 6))

# 🔸 KMeans sonucu
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50, alpha=0.7, edgecolors="k", cmap="tab10")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="red", s=200, marker="X", label="Merkezler")
plt.title("KMeans - Müşteri Segmentasyonu")
plt.xlabel("Gelir (Income)")
plt.ylabel("Harcama Skoru (Spending Score)")
plt.legend()

# 🔸 Dendrogram (Hiyerarşik Kümeleme)
plt.subplot(1, 2, 2)
linkage_matrix = linkage(X, method="ward")
dendrogram(linkage_matrix, truncate_mode="lastp", p=12, leaf_rotation=45., leaf_font_size=10.)
plt.title("Dendrogram - Müşteri Segmentasyonu")
plt.xlabel("Veri Noktaları")
plt.ylabel("Uzaklık")

plt.tight_layout()
plt.show()
