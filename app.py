import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

url= 'k3_best_model_maize_diseases.weights.h5'
# Charger le modèle avec vérification
@st.cache_resource
def load_model_from_file():
    model = load_model(url)
    return model

# Charger et afficher le modèle
st.title("🌽 Détection des Maladies des Feuilles de Maïs")

with st.spinner('Chargement du modèle...'):
    model = load_model_from_file()
if model:
    st.success("Modèle chargé avec succès !")
else:
    st.error("Impossible de charger le modèle.")

# Widgets pour charger une image
uploaded_file = st.file_uploader("Téléchargez une image de la feuille", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Ou prenez une photo via la webcam")

# Traitement et prédiction
if uploaded_file or camera_input:
    image = Image.open(uploaded_file or camera_input)
    st.image(image, caption="Image chargée", use_column_width=True)

    # Prétraitement
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    try:
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Classes de maladies
        disease_names = [
            "Cercospora Leaf Spot (Tâche grise)",
            "Common Rust (Rouille commune)",
            "Northern Leaf Blight (Brûlure du nord)",
            "Feuille saine"
        ]
        result = disease_names[predicted_class]

        # Afficher le résultat
        st.markdown(f"**Résultat :** {result}")
        st.progress(confidence)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
