#!/usr/bin/env python3
import os
import io
import uuid
import logging
import numpy as np
import cv2
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
from PIL import Image

# NAFNet imports
from basicsr.models import create_model
from src.lib.basicsr.utils import img2tensor as _img2tensor, tensor2img
from src.lib.basicsr.utils.options import parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nafnet-api")

# Initialize FastAPI app
app = FastAPI(
    title="NAFNet Image Restoration API",
    description="API for image restoration using NAFNet model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global variables
MODEL = None
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

def load_model():
    """Load the NAFNet model once at startup"""
    global MODEL
    if MODEL is None:
        logger.info("Loading NAFNet model...")
        try:
            # Use the same configuration as in run_inference.py
            opt_path = 'src/lib/options/test/REDS/NAFNet-width64.yml'
            opt = parse(opt_path, is_train=False)
            opt['dist'] = False
            # Force CPU mode
            opt['num_gpu'] = 0
            MODEL = create_model(opt)
            logger.info("NAFNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    return MODEL

def img2tensor(img, bgr2rgb=False, float32=True):
    """Convert image to tensor"""
    img = img.astype(np.float32) / 255.0
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def process_image(model, img_np):
    """Process the image using NAFNet model"""
    # Convert to RGB if needed (OpenCV loads in BGR)
    if img_np.shape[2] == 3:  # Color image
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np
    
    # Convert to tensor
    img_tensor = img2tensor(img_rgb)
    
    # Run inference
    model.feed_data(data={'lq': img_tensor.unsqueeze(0)})
    model.test()
    
    # Get result
    visuals = model.get_current_visuals()
    output_img = tensor2img([visuals['result']])
    
    # Convert back to RGB format for returning
    if output_img.shape[2] == 3:  # Color image
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    return output_img

@app.get("/", response_class=HTMLResponse)
async def root():
    """Return a simple HTML page with an upload form"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>NAFNet Image Restoration</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                form {
                    margin: 20px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                input[type="file"] {
                    margin: 10px 0;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                .images {
                    display: flex;
                    flex-wrap: wrap;
                    margin-top: 20px;
                }
                .image-container {
                    margin: 10px;
                    text-align: center;
                }
                img {
                    max-width: 100%;
                    max-height: 400px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }
            </style>
        </head>
        <body>
            <h1>NAFNet Image Restoration</h1>
            <p>Upload an image to process it with NAFNet for image restoration.</p>
            <form action="/process/" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Process Image">
            </form>
            <div id="result" class="images"></div>
            <script>
                const form = document.querySelector('form');
                const result = document.getElementById('result');
                
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    
                    const formData = new FormData(form);
                    const file = formData.get('file');
                    
                    if (!file || file.size === 0) {
                        alert('Please select an image file');
                        return;
                    }
                    
                    // Display the original image
                    result.innerHTML = `
                        <div class="image-container">
                            <h3>Original Image</h3>
                            <img src="${URL.createObjectURL(file)}" alt="Original">
                        </div>
                        <div class="image-container">
                            <h3>Processing...</h3>
                        </div>
                    `;
                    
                    try {
                        const response = await fetch('/process/', {
                            method: 'POST',
                            body: formData,
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error ${response.status}`);
                        }
                        
                        const blob = await response.blob();
                        const processedImageUrl = URL.createObjectURL(blob);
                        
                        result.innerHTML = `
                            <div class="image-container">
                                <h3>Original Image</h3>
                                <img src="${URL.createObjectURL(file)}" alt="Original">
                            </div>
                            <div class="image-container">
                                <h3>Processed Image</h3>
                                <img src="${processedImageUrl}" alt="Processed">
                                <p><a href="${processedImageUrl}" download="nafnet_processed.png">Download</a></p>
                            </div>
                        `;
                    } catch (error) {
                        console.error('Error:', error);
                        result.innerHTML += `<p style="color: red;">Error: ${error.message}</p>`;
                    }
                });
            </script>
        </body>
    </html>
    """

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    load_model()

@app.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Process an uploaded image with NAFNet
    
    - **file**: An image file upload
    
    Returns the processed image as a response
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")
    
    try:
        # Load the model if not already loaded
        model = load_model()
        
        # Read and convert the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")
        
        # Process the image
        output_img = process_image(model, img)
        
        # Convert the processed image to bytes for returning
        is_success, buffer = cv2.imencode(".png", output_img)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
        
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        
        # Return the processed image
        return StreamingResponse(
            io_buf, 
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=nafnet_processed.png"}
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if MODEL is None:
        try:
            load_model()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "ok", "model_loaded": MODEL is not None}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

