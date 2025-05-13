# NAFNet API Service

## Project Description

This project provides a web-based API service for image restoration using NAFNet (Nonlinear Activation Free Network), a state-of-the-art image restoration model. The service can restore degraded images affected by blur, noise, or low resolution. NAFNet is designed to be efficient while maintaining high-quality restoration results.

NAFNet is based on the paper [NAFNet: Nonlinear Activation Free Network for Image Restoration](https://arxiv.org/abs/2204.04676).

## Requirements

- Python 3.8+
- PyTorch 1.8+
- FastAPI
- UV (optional, for faster dependency management)
- Other dependencies listed in `requirements.txt`

## Installation Instructions

### Setup with UV (Recommended)

UV is a fast Python package installer and virtual environment manager. It offers significantly faster dependency resolution and installation compared to pip.

#### 1. Install UV

First, you need to install UV:

```bash
# For Linux/macOS
curl -sSf https://install.determinate.systems/uv | sh -s -- -y

# For Windows (using PowerShell)
irm https://install.determinate.systems/uv | iex
```

#### 2. Clone the repository

```bash
git clone https://github.com/harshilvagadiya/nafnet-api.git
cd NAFNet
```

#### 3. Create a virtual environment with UV

```bash
# Create a new virtual environment in a folder named 'venv'
uv venv
```

#### 4. Activate the virtual environment

```bash
# On Linux/macOS
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

#### 5. Install dependencies with UV

```bash
# Install all requirements using UV (much faster than pip)
uv pip install -r requirements.txt
```

When you're done working with NAFNet, you can deactivate the virtual environment:

```bash
deactivate
```

## Downloading Pretrained Model

Due to file size limitations, the pretrained model needs to be downloaded separately. After setting up your environment:

1. Create the directory for the model:
```bash
mkdir -p src/lib/experiments/pretrained_models
```

2. Download the NAFNet-REDS-width64.pth model from [Hugging Face](https://huggingface.co/megvii-research/NAFNet) or other model hosting platforms.

3. Place the downloaded model file in:
```
src/lib/experiments/pretrained_models/NAFNet-REDS-width64.pth
```

This step is required before running the API server.

## How to Run the API Server

To run the NAFNet API server, you need to set the Python path to include the source libraries and then start the API:

```bash
# Run from the project root directory
PYTHONPATH=$PYTHONPATH:src/lib python api.py
```

On Windows:

```bash
set PYTHONPATH=%PYTHONPATH%;src\lib
python api.py
```

The server will start on `http://0.0.0.0:8000` by default. You can access the web interface by navigating to this URL in your browser.

## API Endpoints Documentation

The API provides the following endpoints:

### 1. Web Interface

- **URL:** `/`
- **Method:** `GET`
- **Description:** Provides a user-friendly web interface for uploading and processing images.

### 2. Process Image

- **URL:** `/process/`
- **Method:** `POST`
- **Request:**
  - Content-Type: `multipart/form-data`
  - Body: Form data with a file field named `file` containing the image to process
- **Response:**
  - Content-Type: `image/png`
  - Body: The processed image
- **Description:** Uploads an image for processing with NAFNet and returns the restored image.

### 3. Health Check

- **URL:** `/health`
- **Method:** `GET`
- **Response:**
  - Content-Type: `application/json`
  - Body: `{"status": "ok", "model_loaded": true}` if the service is healthy
- **Description:** Checks if the service is running and the model is loaded correctly.

## Examples of Usage

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:8000`
2. Click the "Choose File" button and select an image to restore
3. Click "Process Image" to start the restoration
4. The original and restored images will be displayed side by side
5. You can download the restored image using the "Download" link


## Troubleshooting Common Issues

### ModuleNotFoundError: No module named 'basicsr'

**Solution:** Make sure you're running the API with the correct PYTHONPATH:

```bash
PYTHONPATH=$PYTHONPATH:src/lib python api.py
```

### CUDA out of memory error

**Solution:** The model is trying to use too much GPU memory. You can:

1. Use a smaller image size
2. Edit `api.py` to use CPU mode (already set by default)
3. If you need GPU mode, reduce batch size or model parameters

### Import errors with torch or other dependencies

**Solution:** Make sure all dependencies are installed correctly:

```bash
pip install -r requirements.txt
```

### Connection refused when accessing the web interface

**Solution:** Check that the API server is running and listening on the correct port:

1. Make sure there's no error in the terminal where you started the server
2. Check if the port 8000 is already in use by another application
3. Try accessing http://127.0.0.1:8000 instead of http://localhost:8000

### Slow processing speed

**Solution:**
1. Processing on CPU is slower than GPU. If available, configure the model to use GPU by modifying line 60 in `api.py`:
   ```python
   opt['num_gpu'] = 1  # Set to number of GPUs available
   ```
2. If using GPU and still slow, try optimizing the model parameters or use a more powerful GPU.

## License

This project is licensed under the terms of the Apache License 2.0.

