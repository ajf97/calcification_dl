import os
from tempfile import NamedTemporaryFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import plotly.express as px
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
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

with st.sidebar:
    st.header("Segmenting calcifications on mammograms")

    width = st.number_input(
        "Image resolution (width)", value=1000, min_value=10, max_value=4000
    )
    threshold = st.number_input(
        "Prediction threshold", value=0.01, min_value=0.0, max_value=1.0, format="%f"
    )

    uploaded_image = st.file_uploader("Upload the input image")
    uploaded_model = st.file_uploader("Upload the model file")
    bt = st.button("Predict")


if bt and uploaded_image and uploaded_model is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Preprocess image
    image_preprocessed = transforms.preprocess(image_array, width=width)

    # Load the model and make the prediction
    with NamedTemporaryFile(dir=".", delete=False) as f:
        f.write(uploaded_model.getbuffer())
        model = initialize_model(f.name)
        f.close()
        os.unlink(f.name)

    pred = predict(model, image_preprocessed)
    predi = pred.copy()
    sm = segmentation_map(pred, threshold=threshold)

    # Apply mask to image
    masked_image = apply_mask(image_preprocessed, sm)

    # Put the images in two columns
    tab1, tab2 = st.tabs(["📈 Output and probability heatmap", "🎭 Segmentation mask"])

    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            fig = plt.imshow(masked_image)
            pr = fig.make_image(renderer=None, unsampled=True)[0]
            fig = px.imshow(pr, aspect="equal")
            fig.update_layout(width=600, height=600)
            col1.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.imshow(predi, aspect="equal", color_continuous_scale="turbo")
            fig.update_layout(width=600, height=600)
            col2.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(sm, aspect="equal", color_continuous_scale="gray")
            fig.update(layout_coloraxis_showscale=False)
            fig.update_layout(width=600, height=600)
            col1.plotly_chart(fig, use_column_width=True, clamp=True)
