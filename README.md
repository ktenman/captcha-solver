# CAPTCHA Solver

## Overview
This project is a web application designed to solve CAPTCHA challenges using machine learning. It utilizes a TensorFlow model to decode CAPTCHA images and offers a solution via a Flask-based web interface or API.

## Features
- **CAPTCHA Recognition**: Uses advanced machine learning techniques to decode CAPTCHAs.
- **Web Interface**: Built with Flask, providing an easy-to-use interface or API for CAPTCHA solving.
- **Model Management**: Includes functionality to download and load a pre-trained TensorFlow model.
- **Image Preprocessing**: Processes CAPTCHA images for optimal recognition by the model.
- **Hardcoded Character Set**: Utilizes a predefined character set for CAPTCHA decoding.

## Installation and Running with Docker
1. **Clone the Repository**: Clone this repository to your local machine.
2. **Build Docker Image**: Run `docker build -t captcha-solver .` to build the Docker image.
3. **Run the Container**: Execute `docker run -p 52525:52525 captcha-solver` to start the application in a Docker container.

   The application will be accessible at `http://localhost:52525`.

## Testing with image from Docker Hub
1. **Pull Docker Image**: Run `docker pull ktenman/captcha-solver:latest` to pull the Docker image.
2. **Run the Container**: Execute `docker run -p 52525:52525 ktenman/captcha-solver:latest` to start the application in a Docker container.

   The application will be accessible at `http://localhost:52525`.

## Usage
- Access the web interface via the URL provided by the Docker container, typically `http://localhost:52525`.
- Follow the on-screen instructions or API documentation for solving CAPTCHAs.

## Training on Google Colab
https://drive.google.com/file/d/1RGOzJSUXnLUtAPMCjgh2QAA5W9uTPEyu/view?usp=sharing

## Contributing
Contributions to the project are welcome. Please ensure to follow the code standards and submit pull requests for any new features or bug fixes.

## License
This project is licensed under [LICENSE]. Please see the LICENSE file for more details.
