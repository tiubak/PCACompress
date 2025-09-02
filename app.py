import streamlit as st
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import os
import io

st.title("PCA Image Compression")
st.subheader("This is the final project for Unsupervised Algorithms in Machine Learning from University of Colorado Boulder")
st.page_link("https://github.com/tiubak/PCACompress", label="https://github.com/tiubak/PCACompress", icon="🌎")

uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Show original file size
    uploaded_file.seek(0, os.SEEK_END)
    original_size_MB = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)
    st.write(f"Original Image Size (compressed, MB): {original_size_MB:.2f}")

    # Convert to grayscale
    grayscale_image = image.convert("L")
    grayscale_array = np.array(grayscale_image)
    st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

    # Show array sizes
    original_array = np.array(image)
    original_array_MB = original_array.nbytes / (1024 * 1024)
    grayscale_array_MB = grayscale_array.nbytes / (1024 * 1024)
    st.write(f"Original Image Array Size (uncompressed, MB): {original_array_MB:.2f}")
    st.write(f"Grayscale Image Array Size (uncompressed, MB): {grayscale_array_MB:.2f}")

    # PCA Compression
    st.write("---")
    st.subheader("PCA Compression")    
    scaled_data = grayscale_array / 255.0
    pca = PCA(n_components=0.95) #keep the 95% variance
    pca.fit(scaled_data)
    n_components = pca.n_components_
    transformed = pca.transform(scaled_data)
    reconstructed = pca.inverse_transform(transformed)
    reconstructed_img = (reconstructed * 255).clip(0, 255).astype(np.uint8)
    reconstructed_pil = Image.fromarray(reconstructed_img)
    st.image(reconstructed_pil, caption=f"Reconstructed Image ({n_components} components: 95% variance)", use_container_width=True)

    # Save reconstructed image to buffer and show size
    buf = io.BytesIO()
    reconstructed_pil.save(buf, format="JPEG")
    buf.seek(0, os.SEEK_END)
    compressed_size_MB = buf.tell() / (1024 * 1024)
    buf.seek(0)
    st.write(f"Reconstructed Image Size (converted to JPG) MB: {compressed_size_MB:.2f}")
else:
    st.info("Please upload an image to begin.")

st.info("**This demo made with **Google Gemini 2.5 PRO** to reply questions about how to user streamlit and the notebook create as delivery.**")
