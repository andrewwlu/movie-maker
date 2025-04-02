import cv2
import numpy as np
from pathlib import Path
import re

# Input and output directories
input_dir = Path("../output/1_stitched")
output_base = Path("../output/2a_enhanced_channels")

# Create output directories for each channel
channel_dirs = ["DIC", "GFP", "Ruby"]
for channel in channel_dirs:
    (output_base / channel).mkdir(parents=True, exist_ok=True)

def enhance_contrast(img, alpha=1.5, beta=30):
    """Enhance contrast and brightness"""
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

def enhance_dic(gray):
    """Create extremely subtle DIC outline"""
    # Stronger blur for more subtle edges
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.subtract(gray, blurred)
    
    # Extreme contrast reduction
    p30, p70 = np.percentile(gray, (30, 70))  # Narrower range
    processed = np.clip((gray - p30) * (100.0 / (p70 - p30)), 0, 255).astype(np.uint8)
    
    # Very subtle edge blend
    processed = cv2.addWeighted(processed, 0.8, edges, 0.2, 0)
    
    # Drastically reduce intensity
    processed = enhance_contrast(processed, alpha=0.3, beta=0)
    
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

def enhance_gfp(gray):
    """Enhance GFP with binary thresholding"""
    # Strong initial contrast enhancement (reduced to 80%)
    processed = cv2.convertScaleAbs(gray, alpha=5.2, beta=4.4)  # Was 6.5 and 5.5
    
    # Noise reduction
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    
    # Binary threshold (reduced to 80%)
    _, binary = cv2.threshold(processed, 15, 255, cv2.THRESH_BINARY)  # Was 18.5
    
    # Minimal noise reduction
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Create green channel
    zeros = np.zeros_like(binary)
    return cv2.merge([zeros, binary, zeros])

def enhance_ruby(gray):
    """Enhance Ruby with binary thresholding"""
    # Strong initial contrast enhancement (reduced to 80%)
    processed = cv2.convertScaleAbs(gray, alpha=5.2, beta=4.4)  # Was 6.5 and 5.5
    
    # Noise reduction
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    
    # Binary threshold (reduced to 80%)
    _, binary = cv2.threshold(processed, 15, 255, cv2.THRESH_BINARY)  # Was 18.5
    
    # Minimal noise reduction
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Create red channel
    zeros = np.zeros_like(binary)
    return cv2.merge([zeros, zeros, binary])

# Get all stitched images
stitched_images = list(input_dir.glob("well_*_channel_*.png"))

# Process each image
for img_path in stitched_images:
    # Parse well and channel from filename
    match = re.match(r"well_(\d+)_channel_(.+)\.png", img_path.name)
    if not match:
        continue
    
    well, channel = match.groups()
    
    # Read and convert to grayscale
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Process based on channel type
    if "DIC" in channel:
        processed = enhance_dic(gray)
        output_dir = output_base / "DIC"
    elif "GFP" in channel:
        processed = enhance_gfp(gray)
        output_dir = output_base / "GFP"
    else:  # Ruby
        processed = enhance_ruby(gray)
        output_dir = output_base / "Ruby"
    
    # Save processed image
    output_path = output_dir / f"well_{well}.png"
    cv2.imwrite(str(output_path), processed)
    print(f"Processed {channel} for well {well}")

print(f"Enhanced channel images saved to {output_base}") 