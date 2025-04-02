import cv2
import numpy as np
from pathlib import Path

# Input and output directories
input_base = Path("../output/2a_enhanced_channels")
output_dir = Path("../output/2b_merged")
output_dir.mkdir(parents=True, exist_ok=True)

# Fluorescence colors (in BGR format for OpenCV)

### REALLY GOOD OPTION
RUBY_COLOR = (71, 160, 67)
GFP_COLOR = (54, 67, 244)

def enhance_contrast(img, alpha=1.5, beta=30):
    """Enhance contrast and brightness"""
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

# Get all well numbers from GFP directory
gfp_files = list((input_base / "GFP").glob("well_*.png"))
well_numbers = [f.stem.split('_')[1] for f in gfp_files]

# Process each well
for well in well_numbers:
    # Read enhanced channels
    gfp_path = input_base / "GFP" / f"well_{well}.png"
    ruby_path = input_base / "Ruby" / f"well_{well}.png"
    
    # Check if both channels exist
    if not all(p.exists() for p in [gfp_path, ruby_path]):
        print(f"Skipping well {well} - missing some channels")
        continue
    
    # Read images and extract masks
    gfp_img = cv2.imread(str(gfp_path), cv2.IMREAD_GRAYSCALE)
    ruby_img = cv2.imread(str(ruby_path), cv2.IMREAD_GRAYSCALE)
    
    # Start with black background
    merged = np.zeros((gfp_img.shape[0], gfp_img.shape[1], 3), dtype=np.uint8)
    
    # Apply colors to each channel
    merged[gfp_img > 0] = GFP_COLOR
    merged[ruby_img > 0] = RUBY_COLOR
    
    # Save merged image
    output_path = output_dir / f"well_{well}_merged.png"
    cv2.imwrite(str(output_path), merged)
    print(f"Merged channels for well {well}")

print(f"Merged images saved to {output_dir}") 