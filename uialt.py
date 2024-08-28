import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import load_model
from simple_multi_unet_model import multi_unet_model 
from keras.utils import normalize
import os
import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_and_preprocess_images():
    train_images = []
    for img_path in glob.glob("128_patches/images/*.tif"):
        img = cv2.imread(img_path, 0)
        train_images.append(img)

    train_images = np.array(train_images)

    train_masks = []
    for mask_path in glob.glob("128_patches/masks/*.tif"):
        mask = cv2.imread(mask_path, 0)
        train_masks.append(mask)

    train_masks = np.array(train_masks)

    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    train_images = np.expand_dims(train_images, axis=3)
    train_images = normalize(train_images, axis=1)

    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size=0.10, random_state=0)
    X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size=0.2, random_state=0)

    y_train_cat = to_categorical(y_train, num_classes=4).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], 4))
    y_test_cat = to_categorical(y_test, num_classes=4).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], 4))

    return X_test, y_test, y_test_cat

def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    return y_pred_argmax

def preprocess_user_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    
    img_resized = cv2.resize(img, (128, 128))
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_resized = normalize(img_resized, axis=1)
    
    return np.expand_dims(img_resized, axis=0)

st.set_page_config(page_title="Image Segmentation Demonstration", layout="wide")

st.markdown("<h1 style='text-align: center; font-weight: bold;'>Demonstration of Image Segmentation</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; font-size:18px;'>
This is a demonstration of image segmentation techniques. Image segmentation is the process of partitioning an image into multiple segments or regions, each of which is more meaningful and easier to analyze. The goal is to simplify or change the representation of an image into something that is more meaningful and easier to understand.
</p>
""", unsafe_allow_html=True)

if st.button('Reload'):
    st.experimental_rerun()

X_test, y_test, y_test_cat = load_and_preprocess_images()

model = load_model('test.hdf5')

y_pred_argmax = make_predictions(model, X_test)

test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth = y_test[test_img_number]
test_img_norm = test_img[:, :, 0][:, :, None]
test_img_input = np.expand_dims(test_img_norm, 0)
prediction = model.predict(test_img_input)
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h3 style='text-align: center;'>Testing Image</h3>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots()
    ax1.imshow(test_img[:, :, 0], cmap='gray')
    ax1.axis('off')
    st.pyplot(fig1)

with col2:
    st.markdown("<h3 style='text-align: center;'>Testing Label</h3>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots()
    ax2.imshow(ground_truth[:, :, 0], cmap='jet')
    ax2.axis('off')
    st.pyplot(fig2)

with col3:
    st.markdown("<h3 style='text-align: center;'>Prediction on Test Image</h3>", unsafe_allow_html=True)
    fig3, ax3 = plt.subplots()
    ax3.imshow(predicted_img, cmap='jet')
    ax3.axis('off')
    st.pyplot(fig3)

st.markdown("<h2 style='text-align: center;'>Upload Your Image for Segmentation</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    user_img = preprocess_user_image(uploaded_file)
    
    user_prediction = model.predict(user_img)
    user_predicted_img = np.argmax(user_prediction, axis=3)[0, :, :]
    
    st.markdown("<h3 style='text-align: center;'>User Uploaded Image and Prediction</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align: center;'>Uploaded Image</h4>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots()
        ax4.imshow(user_img[0, :, :, 0], cmap='gray')
        ax4.axis('off')
        st.pyplot(fig4)
    
    with col2:
        st.markdown("<h4 style='text-align: center;'>Predicted Mask</h4>", unsafe_allow_html=True)
        fig5, ax5 = plt.subplots()
        ax5.imshow(user_predicted_img, cmap='jet')
        ax5.axis('off')
        st.pyplot(fig5)

