# Açıklama: Bu kod, sklearn kütüphanesinin göğüs kanseri veri setini kullanarak KNN algoritması ile sınıflandırma ve regresyon işlemlerini gerçekleştirir. 
# Ayrıca K değerine göre doğruluk analizi ve regresörlerin (uniform, distance) görselleştirilmesini içerir.
# Gerekli kütüphaneler: pip install scikit-learn pandas matplotlib numpy

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

X = cancer.data  # features
y = cancer.target  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasını eğitir

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

"""
    KNN: Hyperparameter = K
        K: 1,2,3 ... N
        Accuracy: %A, %B, %C ....
"""
accuracy_values = []
k_values = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker="o", linestyle="-")
plt.title("K değerine göre doğruluk")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)
plt.show()

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# Gürültü ekleme
y[::5] += 1 * (0.5 - np.random.rand(8))

T = np.linspace(0, 5, 500)[:, np.newaxis]

fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # Boyut artırıldı

for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)

    ax = axes[i]
    ax.scatter(X, y, color="green", label="Veri", alpha=0.7)
    ax.plot(T, y_pred, color="blue", label="Tahmin", linewidth=2)
    ax.set_title(f"KNN Regresör (weights = {weight})", fontsize=12, pad=10)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.tight_layout(pad=3.0)
plt.show()
