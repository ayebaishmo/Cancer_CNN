import streamlit as st
from PIL import Image
import numpy as np
import json
import os
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

st.title(" Skin Lesion Classifier")
st.write("Upload a dermoscopic image to predict the type of skin lesion.")

model_path = hf_hub_download(repo_id="Ishmo-plug/cnn-skin-lesion-model", filename="cnn_skin_lesion_model.h5")
model = load_model(model_path)

label_map_path = hf_hub_download(repo_id="your-username/cnn-skin-lesion-model", filename="label_map.json")
with open(label_map_path, "r") as f:
    label_map = json.load(f)
idx_to_label = {int(v): k for k, v in label_map.items()}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(image)
    input_array = input_tensor.numpy()
    input_array = np.expand_dims(np.transpose(input_array, (1, 2, 0)), axis=0)

    prediction = model.predict(input_array)
    predicted_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_label = idx_to_label[predicted_idx]
    st.markdown(f"### Prediction: **{predicted_label}**")
    st.markdown(f"Confidence: **{confidence:.2f}**")
