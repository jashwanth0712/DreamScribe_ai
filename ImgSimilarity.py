import numpy as np
from PIL import Image
import base64
from io import BytesIO

def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decodes a base64 string to an image (numpy array).
    :param base64_str: Base64-encoded image string
    :return: Image as a NumPy array
    """
    # Decode the base64 string
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    
    # Convert the image to grayscale (if it's not already)
    img = img.convert("L")
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    return img_array

def threshold_image(img_array: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Applies a threshold to the image: values >= threshold are set to 1, and < threshold to 0.
    :param img_array: Grayscale image as a NumPy array
    :param threshold: The threshold value (default is 128)
    :return: Binary thresholded image as a NumPy array
    """
    # Apply thresholding
    return np.where(img_array >= threshold, 1, 0)

def similarity(img1_base64: str, img2_base64: str) -> float:
    """
    Given two base64-encoded images, return the percentage similarity based on a binary thresholding comparison.
    :param img1_base64: Base64 string for the first image
    :param img2_base64: Base64 string for the second image
    :return: Similarity percentage (0.0 to 100.0)
    """
    # Decode and threshold both images
    img1_array = decode_base64_image(img1_base64)
    img2_array = decode_base64_image(img2_base64)
    
    # Apply thresholding to both images
    img1_bin = threshold_image(img1_array)
    img2_bin = threshold_image(img2_array)

    # Ensure both images are of the same size
    if img1_bin.shape != img2_bin.shape:
        raise ValueError("Both images must have the same dimensions")

    # Compute the number of matching pixels
    matching_pixels = np.sum(img1_bin == img2_bin)

    # Calculate the total number of pixels
    total_pixels = img1_bin.size

    # Calculate similarity percentage
    similarity_percentage = (matching_pixels / total_pixels) * 100

    return similarity_percentage

