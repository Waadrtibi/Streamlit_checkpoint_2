import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle et les colonnes
model = joblib.load("financial_inclusion_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("💳 Prédiction d'Inclusion Financière en Afrique")

# Interface utilisateur
input_data = {}

for col in model_columns:
    val = st.text_input(f"{col}", "")
    input_data[col] = val

if st.button("Prédire l’accès au compte bancaire"):
    # Créer un DataFrame d’entrée
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype(float)  # conversion des champs
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        result = "✅ A un compte bancaire" if prediction == 1 else "❌ N’a pas de compte bancaire"
        st.success(f"Résultat : {result} (Probabilité : {proba:.2f})")
    except ValueError as e:
        st.error(f"Erreur d’entrée : {e}")
