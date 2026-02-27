# 🔹 Açıklama: Bu kod, Iris veri seti üzerinde Naive Bayes (GaussianNB) sınıflandırıcısını kullanarak modeli eğitir ve test sonuçlarını sınıflandırma raporu olarak ekrana yazdırır.
# 🔹 Gerekli pip paketleri: pip install scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Veri setini yükle
iris = load_iris()

# Özellikler ve etiketler
ozellikler = iris.data
etiketler = iris.target

# Eğitim ve test verisine ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(ozellikler, etiketler, test_size=0.2, random_state=42)

# Naive Bayes modeli oluştur ve eğit
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_egitim, y_egitim)

# Test verisiyle tahmin yap
y_tahmin = naive_bayes_model.predict(X_test)

# Sonuçları yazdır
print(classification_report(y_test, y_tahmin))
