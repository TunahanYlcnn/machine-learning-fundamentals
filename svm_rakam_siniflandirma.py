# 🔹 Açıklama: Bu kod, el yazısı rakamları içeren "Digits" veri seti üzerinde Destek Vektör Makineleri (SVM) sınıflandırıcısını kullanarak modeli eğitir, test eder ve sonuçları sınıflandırma raporu olarak ekrana yazdırır. Ayrıca veri setinden örnek görseller gösterir.
# 🔹 Gerekli pip paketleri: pip install scikit-learn matplotlib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Veri setini yükle
rakam_veri = load_digits()

# İlk 10 örneği görselleştir
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5),
                         subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(rakam_veri.images[i], cmap="binary", interpolation="nearest")
    ax.set_title(rakam_veri.target[i])

plt.show()

# Özellikler ve etiketler
ozellikler = rakam_veri.data
etiketler = rakam_veri.target

# Eğitim ve test verisine ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(
    ozellikler, etiketler, test_size=0.2, random_state=42)

# SVM modeli oluştur ve eğit
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_egitim, y_egitim)

# Test verisiyle tahmin yap
y_tahmin = svm_model.predict(X_test)

# Sonuçları yazdır
print(classification_report(y_test, y_tahmin))
