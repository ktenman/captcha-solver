import base64
import flask
import logging
import numpy as np
import os
import requests
import tensorflow as tf
import time
import zipfile
from pathlib import Path
from tensorflow.keras import layers

app = flask.Flask(__name__)
app.logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG for detailed logs

MODEL_PATH = "saved_model999"
IMG_WIDTH = 65
IMG_HEIGHT = 25
HARDCODED_CHAR_SET = ['2', '3', '4', '5', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'K', 'L', 'M', 'N', 'P',
                      'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

sample = "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"

def download_and_unzip(file_name, extract_to):
    app.logger.debug(f"Starting download_and_unzip for {file_name}")
    url = f"https://github.com/ktenman/captcha-solver/raw/main/{file_name}.zip"
    app.logger.info(f"Downloading from {url}")
    response = requests.get(url)
    if response.status_code != 200:
        app.logger.error(f"Failed to download from {url}")
        return False

    zip_file_path = str(extract_to) + ".zip"

    with open(zip_file_path, "wb") as file:
        file.write(response.content)

    app.logger.info(f"Unzipping {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_file_path)
    app.logger.info(f"Unzipped and removed {zip_file_path}")
    return True


def load_model(model_path):
    app.logger.debug(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    app.logger.debug("Model loaded successfully")
    return model


def create_lookup_layers_with_hardcoded_chars():
    app.logger.debug("Creating lookup layers with hardcoded characters")
    char_to_num = layers.StringLookup(vocabulary=HARDCODED_CHAR_SET, mask_token=None)
    num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True, mask_token=None)
    return char_to_num, num_to_char


def process_image_file(img_path, img_height, img_width):
    app.logger.debug(f"Processing image file {img_path}")
    img_path = str(img_path)  # Convert PosixPath to string
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img


def decode_predictions(pred, max_length, num_to_char):
    app.logger.debug("Decoding predictions")
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    return [tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace("[UNK]", "").strip() for res in
            results]


model = None
char_to_num = None
num_to_char = None
max_length = None


def initialize_model_and_mappings():
    global model, char_to_num, num_to_char, max_length
    app.logger.debug("Initializing model and mappings")

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        app.logger.debug(f"Model path {model_path} does not exist. Attempting to download and unzip the model.")
        if not download_and_unzip("saved_model999", model_path):
            app.logger.warning("Failed to download model, using the local model instead.")
            # Extract the local saved_model999.zip
            with zipfile.ZipFile("saved_model999.zip", 'r') as zip_ref:
                zip_ref.extractall(model_path)

    model = load_model(MODEL_PATH)
    char_to_num, num_to_char = create_lookup_layers_with_hardcoded_chars()
    max_length = 4  # Example max length, set according to your captcha length
    app.logger.debug("Model and mappings initialized successfully")


# Function to predict a single image
def predict_single_image(image_path):
    global model, char_to_num, num_to_char, max_length
    app.logger.debug(f"Predicting single image: {image_path}")
    processed_image = process_image_file(image_path, IMG_HEIGHT, IMG_WIDTH)
    preds = model.predict(processed_image)
    decoded_preds = decode_predictions(preds, max_length, num_to_char)
    return decoded_preds[0]


def predict_single_image_from_bytes(image_bytes):
    global model, char_to_num, num_to_char, max_length
    app.logger.debug("Predicting single image from bytes")
    processed_image = process_image_bytes(image_bytes, IMG_HEIGHT, IMG_WIDTH)
    preds = model.predict(processed_image)
    decoded_preds = decode_predictions(preds, max_length, num_to_char)
    return decoded_preds[0]


def process_image_bytes(image_bytes, img_height, img_width):
    img = tf.io.decode_png(image_bytes, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

def load():
    image_bytes = base64.b64decode(sample)
    prediction = predict_single_image_from_bytes(image_bytes)
    print(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.perf_counter_ns()

    try:
        data = flask.request.json
        uuid = data.get('uuid')
        base64_image = data.get('base64_image')

        if not uuid or not base64_image:
            app.logger.info("Missing UUID or base64_image")
            return flask.jsonify({'error': 'Missing UUID or base64_image'}), 400

        image_bytes = base64.b64decode(base64_image)
        prediction = predict_single_image_from_bytes(image_bytes)

        end_time = time.perf_counter_ns()
        duration_ns = end_time - start_time
        duration_ms = duration_ns / 1_000_000
        formatted_duration = f"{duration_ms:0.1f} ms"

        return flask.jsonify({'uuid': uuid, 'predicted_text': prediction, 'duration': formatted_duration}), 200
    except Exception as e:
        end_time = time.perf_counter_ns()
        duration_ns = end_time - start_time
        duration_ms = duration_ns / 1_000_000
        formatted_duration = f"{duration_ms:0.1f} ms"

        app.logger.error(f"Error: {str(e)}, duration={formatted_duration}")

        return flask.jsonify({'error': str(e), 'duration': formatted_duration}), 500


if __name__ == "__main__":
    initialize_model_and_mappings()
    load()  # Explicitly call the load function here
    app.run(port=52525, debug=True, host='0.0.0.0')

