import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from icecream import ic
from loguru import logger
from PIL import Image

import models.unet_base as unet_base
import preprocessing.first_experiments.image_transforms as imgt
import preprocessing.first_experiments.target_transforms as tgt

st.header("Segmentación de calcificaciones en mamografías")
st.write(
    "Elige la imagen de una mamografía para obtener el correspondiente mapa de segmentación."
)

uploaded_file = st.file_uploader("Sube la imagen")


def make_prediction(img, model, postprocess, device):
    model.eval()
    img = img.type(torch.FloatTensor)
    img = img[None, ...]
    x = img.to(device)  # to torch, send to device

    with torch.no_grad():
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs

    return result


def predict(image):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = unet_base.UNet(
        in_channels=1,
        out_channels=2,
        n_blocks=4,
        start_filters=32,
        activation="relu",
        normalization="batch",
        conv_mode="same",
        dim=2,
    ).to(device)

    model_path = "../src/models/cbis_ddsm_unet_base.pt"
    torch.load(model_path, map_location=device)

    return make_prediction(image, model, tgt.PostProcess(), device)


def apply_mask(img, mask):
    red = np.array([255, 0, 0], dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    masked_img = np.where(mask[..., None], red, img)
    output = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

    return output


if uploaded_file is not None:
    # Load image and convert to numpy array
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Imagen de entrada", use_column_width=True)

    image_array = np.array(image)

    # Preprocess image
    image_preprocessing_pipeline = T.Compose([imgt.NormalizeMinMax(), T.ToTensor()])
    img_preprocessed = image_preprocessing_pipeline(image_array)

    # Make prediction and display segmentation map
    segmentation_map = predict(img_preprocessed)
    st.image(segmentation_map, caption="Mapa de segmentación", use_column_width=True)

    # Apply mask to image
    masked_image = apply_mask(image_array, segmentation_map)
    st.image(masked_image, caption="Calcificaciones detectadas", use_column_width=True)
