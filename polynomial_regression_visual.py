# Bu kod, rastgele oluşturulmuş veriler üzerinde 2. dereceden polinomsal regresyon modeli kurarak
# verinin doğrusal olmayan ilişkisini görselleştirir.
# pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Rastgele veri oluşturma
X = 4 * np.random.rand(100, 1)
y = 2 + 3 * X**2 + 2 * np.random.rand(100, 1)  # y = 2 + 3x² + gürültü

# Polinom özellikleri oluştur (2. derece)
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)

# Modeli eğit
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Veri noktalarını çiz
plt.scatter(X, y, color="blue", label="Gerçek Veriler")

# Tahminleri hesapla ve çiz
X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color="red", label="Polinomsal Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polinomsal Regresyon Modeli")
plt.legend()
plt.show()
