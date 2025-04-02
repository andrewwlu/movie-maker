import cv2
import numpy as np
from pathlib import Path

# Input and output directories
input_dir = Path("../output/2b_merged")
output_dir = Path("../output/3_plate_layout")
output_dir.mkdir(parents=True, exist_ok=True)

def add_well_label(img, well_num):
    """Add large, clear well number label to image"""
    # Create a copy to avoid modifying original
    labeled = img.copy()
    
    # Parameters for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 4
    color = (255, 255, 255)  # White
    
    # Get text size
    text = f"Well {well_num}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text in top-left corner with padding
    padding = 20
    x = padding
    y = text_height + padding
    
    # Add black background for better visibility
    cv2.putText(labeled, text, (x+2, y+2), font, font_scale, (0,0,0), thickness+2)
    # Add white text
    cv2.putText(labeled, text, (x, y), font, font_scale, color, thickness)
    
    return labeled

def create_plate_layout():
    """Create 24-well plate layout with white borders"""
    # Read one image to get dimensions
    sample_files = list(input_dir.glob("well_*_merged.png"))
    if not sample_files:
        raise ValueError("No merged images found in input directory")
    
    sample_img = cv2.imread(str(sample_files[0]))
    h, w = sample_img.shape[:2]
    
    # Define border width
    border = 20  # White border width
    
    # Calculate dimensions for full plate (4 rows x 6 columns)
    plate_h = 4 * h + 5 * border  # 4 wells + 5 borders (edges + between wells)
    plate_w = 6 * w + 7 * border  # 6 wells + 7 borders (edges + between wells)
    
    # Create white canvas
    plate = np.full((plate_h, plate_w, 3), 255, dtype=np.uint8)
    
    # Create well numbers array (top-to-bottom, right-to-left)
    # Initialize with reshape(6,4) for 6 columns, 4 rows, then transpose
    well_positions = np.arange(24).reshape(6, 4).T  # This gives us columns of [0,1,2,3], [4,5,6,7], etc.
    well_positions = well_positions[:, ::-1]  # Flip the columns to start from right
    
    # Place each well image
    for row in range(4):
        for col in range(6):
            well_num = well_positions[row, col]
            well_file = input_dir / f"well_{well_num}_merged.png"
            
            if well_file.exists():
                # Calculate position for this well
                x = col * (w + border) + border
                y = row * (h + border) + border
                
                # Read and add label to image
                img = cv2.imread(str(well_file))
                labeled_img = add_well_label(img, well_num)
                
                # Place in plate
                plate[y:y+h, x:x+w] = labeled_img
            else:
                print(f"Warning: Missing image for well {well_num}")
    
    # Save the plate layout
    output_path = output_dir / "plate_layout.png"
    cv2.imwrite(str(output_path), plate)
    print(f"Saved plate layout to {output_path}")

if __name__ == "__main__":
    create_plate_layout() 