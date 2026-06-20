import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ── Charger les données ─────────────────────────────
train = pd.read_csv("data/train/train.csv")
val   = pd.read_csv("data/val/val.csv")

print(f"Train: {len(train)} | Val: {len(val)}")

# ── Vectorisation TF-IDF ────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, analyzer='char_wb', ngram_range=(2,4))

X_train = vectorizer.fit_transform(train['Question'])
X_val   = vectorizer.transform(val['Question'])

y_train = train['Category']
y_val   = val['Category']

# ── Entraînement ────────────────────────────────────
print("\nEntraînement en cours...")
model = LogisticRegression(max_iter=1000, C=5)
model.fit(X_train, y_train)

# ── Évaluation ──────────────────────────────────────
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"\nAccuracy : {acc:.4f} ({acc*100:.2f}%)")
print("\nRapport détaillé :")
print(classification_report(y_val, y_pred))

# ── Sauvegarder le modèle ───────────────────────────
os.makedirs("nlp/models", exist_ok=True)
joblib.dump(model, "nlp/models/intent_model.pkl")
joblib.dump(vectorizer, "nlp/models/vectorizer.pkl")

print("\nModèle sauvegardé ✅")