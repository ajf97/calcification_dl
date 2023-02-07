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
from train.utils import apply_mask, segmentation_map

st.header("Segmenting calcifications on mammograms")
st.write("Choose the input image to get the segmentation map")

uploaded_file = st.file_uploader("Upload image")


if uploaded_file is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Input image", use_column_width=True)

    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Preprocess image
    image_preprocessed = transforms.preprocess(image_array, width=1500)

    # Load the model and make the prediction
    model = initialize_model(cfg.kios_model)
    pred = predict(model, image_preprocessed)
    sm = segmentation_map(pred, threshold=0.001)

    st.image(sm, caption="Segmentation map", use_column_width=True, clamp=True)

    # Apply mask to image
    masked_image = apply_mask(image_preprocessed, sm)
    st.image(masked_image, caption="output", use_column_width=True, clamp=True)
