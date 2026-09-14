"""
Step 7: Streamlit app -- the minimal web interface required by Section 3.1.

Run with:
    streamlit run app/app.py
"""

import sys
import os

# allow importing model/predict.py regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
from PIL import Image
from model.predict import predict

st.set_page_config(page_title="AgriSmart AI - Crop Disease Detector", page_icon="🌱", layout="centered")

st.title("🌱 AgriSmart AI - Crop Disease Detector")
st.write(
    "Upload a photo of a crop leaf. The model identifies the disease "
    "(or confirms it's healthy) and gives basic precautionary guidance."
)

st.info(
    "**Note on accuracy:** this model is trained on lab-condition images (PlantVillage) "
    "and evaluated on real field photos (PlantDoc). Field accuracy is meaningfully lower "
    "than lab accuracy -- see the model report for the measured gap. This is a known, "
    "documented limitation of lab-trained crop disease models, not a malfunction.",
    icon="ℹ️",
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    # save temporarily so predict() can open it by path
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_upload.jpg")
    image.save(temp_path)

    with st.spinner("Analyzing leaf..."):
        result = predict(temp_path)

    with col2:
        st.subheader("Result")
        class_label = result["class_label"].replace("___", " - ").replace("_", " ")
        st.markdown(f"### {class_label}")
        st.progress(result["confidence"])
        st.write(f"**Confidence:** {result['confidence']*100:.1f}%")

        st.markdown("**Precaution:**")
        st.write(result["precaution"])

    st.markdown("---")
    st.subheader("Other possibilities considered")
    for r in result["top_k"][1:]:
        label = r["class_label"].replace("___", " - ").replace("_", " ")
        st.write(f"- {label} ({r['confidence']*100:.1f}%)")

    os.remove(temp_path)

else:
    st.write("No image uploaded yet. Try a sample leaf photo to see a prediction.")

st.markdown("---")
st.caption(
    "AgriSmart AI | Core task: CV-based crop disease detection | "
    "Trained on PlantVillage, evaluated on PlantDoc (field images)"
)