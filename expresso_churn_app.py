import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import gdown


# =============== PARTIE 1 : Chargement des données ===============
@st.cache_data
def load_data():
    # Téléchargement direct depuis Google Drive
    url = "https://drive.google.com/uc?id=12_KUHr5NlHO_6bN5SylpkxWc-JvpJNWe"
    output = "expresso_churn.csv"
    gdown.download(url, output, quiet=False)

    # Lecture du fichier CSV
    df = pd.read_csv(output)

    # Nettoyage des noms de colonnes (en minuscules, sans espaces)
    df.columns = df.columns.str.strip().str.lower()
    
    return df

# =============== PARTIE 2 : Prétraitement ===============
def preprocess_data(df):
    # Vérifier si 'churn' est présent
    if 'churn' not in df.columns:
        st.error("❌ La colonne 'churn' est introuvable dans le dataset.")
        st.write("🧪 Voici les colonnes disponibles :", df.columns.tolist())
        st.stop()

    # Encodage des colonnes catégorielles
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# =============== PARTIE 3 : Entraînement des modèles ===============
def train_models(df):
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Séparation en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    return results

# =============== PARTIE 4 : Interface utilisateur Streamlit ===============
def main():
    st.title("📊 Prédiction de Churn - Dataset Expresso")

    st.markdown("Cette application utilise 3 algorithmes de Machine Learning pour prédire si un client va résilier (churn) ou non.")

    df = load_data()
    st.write("📄 Aperçu des données brutes :", df.head())

    df_clean = preprocess_data(df)
    st.success("✅ Données prétraitées avec succès.")

    results = train_models(df_clean)
    st.subheader("📈 Précision des modèles :")
    for model_name, score in results.items():
        st.write(f"🔹 {model_name} : {score:.2%}")

# =============== Lancement de l'app ===============
if __name__ == "__main__":
    main()
