# 🔹 Açıklama: make_circles veri setinde DBSCAN algoritmasıyla kümeleri bulur ve sonuçları görselleştirir.
# 🔹 Gerekli pip paketleri: pip install scikit-learn matplotlib

from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# --- Veri seti oluşturma ---
X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.08, random_state=42)

# --- Orijinal veri görselleştirme ---
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=10, color="gray")
plt.title("Orijinal Veri (make_circles)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")

# --- DBSCAN kümeleme ---
dbscan = DBSCAN(eps=0.15, min_samples=15)
cluster_labels = dbscan.fit_predict(X)

# --- Sonuçların görselleştirilmesi ---
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", s=10)
plt.title("DBSCAN Sonuçları")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.colorbar(label="Küme Etiketi")
plt.show()
