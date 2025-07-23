import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv(r"C:\Users\Waad RTIBI\Streamlit_checkpoint_2\Financial_inclusion_dataset.csv")

# Affichage d'informations générales
print(df.info())
print(df.describe(include='all'))
print(df.isnull().sum())

# Supprimer les doublons
df.drop_duplicates(inplace=True)

# Gérer les valeurs manquantes
df.fillna(method='ffill', inplace=True)  # ou df.dropna() selon le contexte

# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Sauvegarde des colonnes (important pour Streamlit)
model_columns = df.drop(columns=['bank_account']).columns  # Exclure la target
X = df[model_columns]
y = df['bank_account']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde du modèle et des colonnes
import joblib
joblib.dump(model, 'financial_inclusion_model.pkl')
joblib.dump(model_columns.tolist(), 'model_columns.pkl')

print("Nettoyage terminé et modèle sauvegardé.")
