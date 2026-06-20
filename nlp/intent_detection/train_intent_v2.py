import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Charger les données ─────────────────────────────
train = pd.read_csv("data/train/train.csv")
val   = pd.read_csv("data/val/val.csv")

# ── Vectorisation TF-IDF ────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, analyzer='char_wb', ngram_range=(2,4))

X_train = vectorizer.fit_transform(train['Question'])
X_val   = vectorizer.transform(val['Question'])

y_train = train['Category']
y_val   = val['Category']

# ── Entraînement avec class_weight balanced ─────────
print("Entraînement en cours...")
model = LogisticRegression(max_iter=1000, C=5, class_weight='balanced')
model.fit(X_train, y_train)

# ── Évaluation ──────────────────────────────────────
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_val, y_pred))

# ── Sauvegarder ────────────────────────────────────
joblib.dump(model, "nlp/models/intent_model_v2.pkl")
joblib.dump(vectorizer, "nlp/models/vectorizer.pkl")
print("Modèle v2 sauvegardé ✅")