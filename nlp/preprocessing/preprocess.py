import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

# ── Charger le dataset ──────────────────────────────
df = pd.read_csv(r"B:\arabic-darija-chatbot\data\raw\MedQA-MA; Question Answering Dataset in Moroccan A\Dataset\MedQA_Ma dataset\MedQA_MA.csv")

print(f"Dataset chargé : {df.shape[0]} lignes")

# ── Nettoyage de base ───────────────────────────────
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)        # espaces multiples
    text = re.sub(r'[^\w\s\u0600-\u06FF؟،.]', '', text)  # garde arabe + ponctuation
    return text

df['Question'] = df['Question'].apply(clean_text)
df['Answer']   = df['Answer'].apply(clean_text)
df['Category'] = df['Category'].str.strip()

print("Nettoyage terminé ✅")
print(f"Catégories : {df['Category'].nunique()} spécialités")

# ── Split 80% / 10% / 10% ──────────────────────────
train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category'])
val, test   = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['Category'])

# ── Sauvegarder ────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/MedQA_clean.csv", index=False, encoding="utf-8-sig")
train.to_csv("data/train/train.csv", index=False, encoding="utf-8-sig")
val.to_csv("data/val/val.csv", index=False, encoding="utf-8-sig")
test.to_csv("data/test/test.csv", index=False, encoding="utf-8-sig")

print(f"\nTrain : {len(train)} lignes")
print(f"Val   : {len(val)} lignes")
print(f"Test  : {len(test)} lignes")
print("\nFichiers sauvegardés ✅")