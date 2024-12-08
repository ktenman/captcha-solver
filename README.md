# CAPTCHA Solver

## Overview
This project is a web application designed to solve CAPTCHA challenges using machine learning. It utilizes a TensorFlow model to decode CAPTCHA images and provides a Flask-based API (served via Uvicorn) for easy integration.

## Features
- **CAPTCHA Recognition**: Leverages a deep learning model (TensorFlow) to accurately decode CAPTCHA images.
- **API Integration**: Exposes a simple, RESTful API endpoint for submitting base64-encoded CAPTCHA images and receiving decoded text.
- **Preprocessing & Model Management**: Preprocessing steps to ensure images are correctly formatted for the model and a mechanism to load and serve a pre-trained model.
- **Fixed Character Set**: Uses a predefined set of characters to interpret CAPTCHA results.

## Installation and Running with Docker

### Building from Source
1. Clone the Repository:
   ```bash
   git clone https://github.com/ktenman/captcha-solver.git
   cd captcha-solver
   ```

2. Build the Docker Image:
   ```bash
   docker build -t captcha-solver .
   ```

3. Run the Container:
   ```bash
   docker run -p 8000:8000 captcha-solver
   ```
   The application will be accessible at http://localhost:8000.

### Using the Pre-Built Image from Docker Hub
1. Pull the Docker Image:
   ```bash
   docker pull ktenman/captcha-solver:latest
   ```

2. Run the Container:
   ```bash
   docker run -p 8000:8000 ktenman/captcha-solver:latest
   ```
   The application will be accessible at http://localhost:8000.

## Usage
Send a POST request to the `/predict` endpoint with a JSON payload containing the base64-encoded CAPTCHA image. The service will return the predicted text along with a confidence score.

Example Request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "imageBase64": "<BASE64_IMAGE>"
      }'
```

## Training on Google Colab
A training notebook and model file are available on Google Drive:
https://drive.google.com/file/d/1RGOzJSUXnLUtAPMCjgh2QAA5W9uTPEyu/view?usp=sharing

## Contributing
Contributions are welcome! Please follow the coding standards and open a pull request for any new features or bug fixes.

## License
This project is licensed under the Apache-2.0 license. See the LICENSE file for more details.