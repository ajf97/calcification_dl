import cv2
import numpy as np
import omegaconf
import streamlit as st
from PIL import Image

cfg = omegaconf.OmegaConf.load("config/config.yaml")

import sys

sys.path.insert(0, cfg.src_path)

from models.fcn import initialize_model, predict
from preprocessing import transforms
from utils import apply_mask, segmentation_map

favicon = Image.open(cfg.favicon)

st.set_page_config(
    page_title="Calc Detector",
    page_icon=favicon,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

with st.sidebar:
    st.header("Segmenting calcifications on mammograms")

    width = st.number_input("Image resolution (width)", min_value=10, max_value=4000)
    threshold = st.number_input(
        "Prediction threshold", min_value=0.0, max_value=1.0, format="%f"
    )

    st.write("Choose the input image to get the segmentation map")
    uploaded_file = st.file_uploader("Upload image")
    bt = st.button("Predict")


if bt and uploaded_file is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Preprocess image
    image_preprocessed = transforms.preprocess(image_array, width=width)

    # Load the model and make the prediction
    model = initialize_model(cfg.kios_model)
    pred = predict(model, image_preprocessed)
    sm = segmentation_map(pred, threshold=threshold)

    # Apply mask to image
    masked_image = apply_mask(image_preprocessed, sm)

    # Put the images in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Output")
        st.image(masked_image, use_column_width=True, clamp=True)

    with col2:
        st.header("Segmentation map")
        st.image(sm, use_column_width=True, clamp=True)
