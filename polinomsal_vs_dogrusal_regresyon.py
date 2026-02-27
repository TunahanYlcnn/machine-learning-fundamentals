# 🔹 Açıklama: California Housing veri seti üzerinde Çok Değişkenli Doğrusal Regresyon ve Polinomsal Regresyon modellerini karşılaştırır.
# 🔹 Gerekli pip paketleri: pip install scikit-learn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Veri setini yükleme ---
housing = fetch_california_housing()
X = housing.data
y = housing.target

# --- Eğitim ve test setlerine ayırma ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Polinomsal özellikler oluşturma ---
poly_feat = PolynomialFeatures(degree=2)
X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.transform(X_test)

# --- Polinomsal regresyon modeli ---
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
poly_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print(f"Polynomial Regression RMSE: {poly_rmse:.4f}")

# --- Çok değişkenli doğrusal regresyon modeli ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print(f"Multi-Variable Linear Regression RMSE: {lin_rmse:.4f}")

# --- Sonuç karşılaştırması ---
improvement = (lin_rmse - poly_rmse) / lin_rmse * 100
print(f"Polinomsal model, doğrusal modele göre yaklaşık %{improvement:.2f} iyileşme sağladı.")
