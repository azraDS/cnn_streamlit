import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
# Modeli yukleyin
model = load_model('trained_model8480.h5')
# Siniflandirma kategorileri
categories = ["Clothes", "Groceries", "Health", "Home",
              "Kitchen", "Office, Toys, Games", "Pet Supplies", "Sports", "Tools"]
# Resmi hazirlama fonksiyonu
def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image
# Streamlit arayuzu
st.title("Resim Siniflandirma Uygulamasi")
uploaded_file = st.file_uploader(
    "Bir resim yukleyin", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yuklenen Resim', use_column_width=True)
    if st.button("Tahmin Et"):
        processed_image = prepare_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)
        predicted_class = categories[np.argmax(prediction)]
        st.write(f"Tahmin Edilen Kategori: {predicted_class}")
