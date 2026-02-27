# Açıklama: Bu kod, Diabetes veri setinde Elastic Net regresyon modelini uygular, GridSearchCV ile en iyi hiperparametreleri bulur ve test veri setinde MSE değerini hesaplar.
# Gerekli kütüphaneler: pip install scikit-learn numpy

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Veri seti
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Elastic Net modeli ve GridSearchCV ile hiperparametre arama
elastic_net = ElasticNet()
elastic_net_param_grid = {
    "alpha": [0.1, 1, 10, 100],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]  # L1/L2 penalty ratio
}

elastic_net_grid_search = GridSearchCV(elastic_net, elastic_net_param_grid, cv=5)
elastic_net_grid_search.fit(X_train, y_train)

print("En iyi parametre:", elastic_net_grid_search.best_params_)
print("En iyi skor:", elastic_net_grid_search.best_score_)

# Test veri setinde tahmin ve MSE hesaplama
best_elastic_net_model = elastic_net_grid_search.best_estimator_
y_pred = best_elastic_net_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
