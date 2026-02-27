# Açıklama: Bu kod, Iris veri setinde Decision Tree sınıflandırma modeli uygular ve GridSearchCV ile K-Fold ve Leave-One-Out (LOO) çapraz doğrulama yöntemlerini kullanarak en iyi hiperparametreleri bulur.
# Gerekli kütüphaneler: pip install scikit-learn numpy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Veri seti
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modeli ve hiperparametre aralığı
tree = DecisionTreeClassifier()
tree_param_dist = {"max_depth": [3, 5, 7]}

# --- K-Fold Grid Search ---
kf = KFold(n_splits=10)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv=kf)
tree_grid_search_kf.fit(X_train, y_train)

print("K-Fold en iyi parametre:", tree_grid_search_kf.best_params_)
print("K-Fold en iyi doğruluk:", tree_grid_search_kf.best_score_)

# --- Leave-One-Out (LOO) Grid Search ---
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv=loo)
tree_grid_search_loo.fit(X_train, y_train)

print("LOO en iyi parametre:", tree_grid_search_loo.best_params_)
print("LOO en iyi doğruluk:", tree_grid_search_loo.best_score_)
