# 🔹 Kalp Hastalığı Tahmini (UCI Heart Disease Dataset)
# Veri kaynağı: https://archive.ics.uci.edu/dataset/45/heart+disease
# Gerekli kütüphaneler:
# pip install ucimlrepo scikit-learn pandas

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# Veri setini ID ile indir (45 numaralı UCI dataset: Heart Disease)
kalp_hastaligi = fetch_ucirepo(id=45)

# Özellikleri ve hedef değişkeni DataFrame olarak birleştir
df = pd.DataFrame(data=kalp_hastaligi.data.features)
df["hedef"] = kalp_hastaligi.data.targets

# Eksik değerleri kontrol et ve sil
if df.isna().any().any():
    df.dropna(inplace=True)
    print("Eksik (NaN) değerler temizlendi.")

# Girdi (X) ve hedef (y) değişkenlerini ayır
X = df.drop(["hedef"], axis=1).values
y = df.hedef.values

# Veriyi eğitim ve test olarak ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Lojistik regresyon modelini oluştur ve eğit
lojistik_model = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=100)
lojistik_model.fit(X_egitim, y_egitim)

# Modelin doğruluk oranını hesapla
dogruluk = lojistik_model.score(X_test, y_test)
print("Lojistik Regresyon Doğruluk Oranı: {:.2f}%".format(dogruluk * 100))
