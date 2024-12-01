from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Initialize the FastAPI app
app = FastAPI()

# Utility functions
def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decodes a base64 string to an image (numpy array).
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img = img.convert("L")  # Convert to grayscale
    return np.array(img)

def threshold_image(img_array: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Applies a threshold to the image: values >= threshold are set to 1, and < threshold to 0.
    """
    return np.where(img_array >= threshold, 1, 0)

def calculate_similarity(img1_base64: str, img2_base64: str) -> float:
    """
    Given two base64-encoded images, return the percentage similarity.
    """
    img1_array = decode_base64_image(img1_base64)
    img2_array = decode_base64_image(img2_base64)
    
    img1_bin = threshold_image(img1_array)
    img2_bin = threshold_image(img2_array)

    if img1_bin.shape != img2_bin.shape:
        raise ValueError("Both images must have the same dimensions")

    matching_pixels = np.sum(img1_bin == img2_bin)
    total_pixels = img1_bin.size
    return (matching_pixels / total_pixels) * 100

# Request model
class SimilarityRequest(BaseModel):
    image1: str
    image2: str

# FastAPI endpoint
@app.post("/similarity_search")
def similarity_search(request: SimilarityRequest):
    """
    Endpoint to calculate the similarity between two base64-encoded images.
    """
    try:
        similarity_percentage = calculate_similarity(request.image1, request.image2)
        return {"similarity": similarity_percentage}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the images.")

# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}
