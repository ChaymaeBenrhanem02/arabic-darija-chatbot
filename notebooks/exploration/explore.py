import pandas as pd

df = pd.read_csv(r"B:\arabic-darija-chatbot\data\raw\MedQA-MA; Question Answering Dataset in Moroccan A\Dataset\MedQA_Ma dataset\MedQA_MA.csv")

# Voir toutes les colonnes
print("Colonnes :", df.columns.tolist())

# Voir quelques exemples complets
print("\nExemples complets :")
pd.set_option('display.max_colwidth', None)
print(df.head(10).to_string())