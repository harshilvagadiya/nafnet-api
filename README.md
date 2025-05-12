# NAFNet Image Restoration API

This repository provides a FastAPI-based web API for image restoration using the [NAFNet](https://github.com/megvii-research/NAFNet) model.

## Features

- **Web Interface**: Simple HTML interface for uploading and processing images
- **REST API**: Endpoints for programmatic image processing
- **CPU Compatible**: Modified to work on systems without NVIDIA GPUs
- **Easy Deployment**: FastAPI with built-in Swagger docs

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- FastAPI

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nafnet-api.git
   cd nafnet-api
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```bash
   python api.py
   ```

The server will start on `http://localhost:8000`.

### API Usage

#### Web Interface

Open your browser and navigate to `http://localhost:8000` to use the web interface.

#### REST API

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

### API Documentation

FastAPI automatically generates documentation for the API. Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Architecture

This API integrates the NAFNet model for image restoration and provides both a web interface and REST API for easy access. The system has been modified to work on CPU, making it compatible with systems that don't have NVIDIA GPUs.

## Image Processing

The NAFNet model is used for high-quality image restoration and enhancement. It can:
- Reduce noise in images
- Improve image clarity
- Enhance image quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NAFNet](https://github.com/megvii-research/NAFNet) - The original image restoration model
- [FastAPI](https://fastapi.tiangolo.com/) - For the web framework

