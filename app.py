# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from generate import generate_digit_images
from model import train_generator
import os

st.set_page_config(page_title="Digit Generator", layout="centered")

st.title("🧠 MNIST Digit Generator")
st.write("Choose a digit (0-9) and generate 5 handwritten-style images!")

# Train model if not already saved
if not os.path.exists("digit_gen.pth"):
    st.info("Training model for the first time... ⏳")
    train_generator()
    st.success("Model trained!")

digit = st.slider("Select Digit", 0, 9, 5)
if st.button("Generate Images"):
    images = generate_digit_images(digit)
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
