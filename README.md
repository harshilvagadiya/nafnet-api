# NAFNet Image Restoration API

This repository provides a FastAPI-based web API for image restoration using the [NAFNet](https://github.com/megvii-research/NAFNet) model.

## Features

- **Web Interface**: Simple HTML interface for uploading and processing images
- **REST API**: Endpoints for programmatic image processing
- **CPU Compatible**: Modified to work on systems without NVIDIA GPUs
- **Easy Deployment**: FastAPI with built-in Swagger docs

## Project Structure

This API is designed to work with the original NAFNet model. It provides a user-friendly interface for using NAFNet for image restoration.

## Setup Options

### Option 1: Setup as standalone API (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/harshilvagadiya/nafnet-api.git
   cd nafnet-api
   ```

2. Download the NAFNet project:
   ```bash
   git clone https://github.com/megvii-research/NAFNet.git nafnet-core
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained models from [NAFNet Models](https://github.com/megvii-research/NAFNet#results-and-pre-trained-models) and place them in the `experiments/pretrained_models/` directory.

5. Start the API server:
   ```bash
   python api.py
   ```

### Option 2: Setup as part of existing NAFNet installation

If you already have the NAFNet repository cloned:

1. Add the API files to your NAFNet installation:
   ```bash
   cd your-nafnet-directory
   git clone https://github.com/harshilvagadiya/nafnet-api.git api-files
   cp api-files/api.py .
   ```

2. Install API dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

3. Start the API server:
   ```bash
   python api.py
   ```

## API Usage

The server will start on `http://localhost:8000`.

### Web Interface

Open your browser and navigate to `http://localhost:8000` to use the web interface.

### REST API

- **Process an image**: `POST /process/`
  - Upload an image file to process it
  - Returns the processed image

Example using cURL:
```bash
curl -X POST "http://localhost:8000/process/" -F "file=@your_image.jpg" --output processed.png
```

- **Health check**: `GET /health`
  - Check if the API and model are running properly

Example:
```bash
curl http://localhost:8000/health
```

## API Documentation

FastAPI automatically generates documentation for the API. Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Image Processing

The NAFNet model is used for high-quality image restoration and enhancement. It can:
- Reduce noise in images
- Improve image clarity
- Enhance image quality

## CPU Support

This API has been modified to work on CPU, making it compatible with systems that don't have NVIDIA GPUs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NAFNet](https://github.com/megvii-research/NAFNet) - The original image restoration model
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework

