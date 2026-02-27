# Açıklama: Bu kod, MNIST veri setinde Linear Discriminant Analysis (LDA) uygular ve 2 boyutlu görselleştirme yapar. 
# Ayrıca Iris veri seti üzerinde PCA ve LDA karşılaştırması görselleştirilir.
# Gerekli kütüphaneler: pip install scikit-learn matplotlib numpy

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# --- MNIST LDA ---
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data
y = mnist.target.astype(int)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="tab10", alpha=0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="Digits")
plt.show()

# --- Iris PCA vs LDA ---
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
colors = ["red", "blue", "green"]

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, label=target_name)
plt.legend()
plt.title("PCA of Iris Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=0.8, label=target_name)
plt.legend()
plt.title("LDA of Iris Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()
