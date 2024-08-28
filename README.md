# Image Segmentation Demonstration App

This Streamlit app demonstrates image segmentation techniques using a U-Net model. It allows users to visualize the segmentation results on pre-processed test images and to upload their own images for segmentation.

## Features

- **Image Segmentation Demonstration**: View segmentation results for randomly selected test images.
- **User Image Upload**: Upload an image to get a segmented output based on the trained U-Net model.

## Installation

To run this app locally, you need to set up a Python environment and install the required dependencies.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/image-segmentation-app.git
    cd image-segmentation-app
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download or place your U-Net model file**:
    Make sure you have the pre-trained U-Net model file named `test.hdf5` in the root directory of the project.

5. **Place your dataset**:
    Ensure you have the dataset with images and masks in the `128_patches/images` and `128_patches/masks` directories respectively.

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Access the app**:
    Open a web browser and go to `http://localhost:8501` to interact with the application.

## Description

### `app.py`

- **`load_and_preprocess_images()`**: Loads and preprocesses images and masks, performs train-test split, and encodes masks.
- **`make_predictions(model, X_test)`**: Uses the U-Net model to make predictions on test images.
- **`preprocess_user_image(uploaded_file)`**: Processes uploaded images to the format required by the model.

### Streamlit Interface

- **Main Page**: Displays a random test image, its ground truth mask, and the predicted segmentation mask.
- **Upload Section**: Allows users to upload their own images for segmentation and view the results.

## Notes

- Ensure that the image paths and model file names in the script match your setup.
- For custom U-Net models, adjust the `simple_multi_unet_model.py` as needed.
- The app uses TensorFlow/Keras, OpenCV, and other libraries; ensure compatibility with your environment.

## Contributing

Feel free to open issues or submit pull requests to improve the app. Contributions are welcome!
