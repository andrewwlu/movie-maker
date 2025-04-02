import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)

def parse_filename(filename):
    
    match = re.match(r"E163_w\d+([A-Za-z0-9\-]+)_s(\d+)\.TIF", filename)
    if match:
        raw_channel, position = match.groups()
        return {
            "channel": raw_channel.replace("AL-", ""),
            "position": int(position),
            "filename": filename
        }
    
    # If no pattern matches, return None or raise an exception
    raise ValueError(f"Could not parse filename: {filename}")

image_dir = Path("../input/images")

# Get all TIF files and parse their info into a dataframe
file_paths = list(image_dir.glob("*.TIF"))
df = pd.DataFrame([parse_filename(fp.name) for fp in file_paths])
df['file_path'] = file_paths
df = df.sort_values(by="position")
print(df.head(20))
print('\n\n\n')

stage_positions_df = pd.read_csv("../input/others/clean_GeneratedStage_fixed_small.csv")
df = pd.merge(df, stage_positions_df, on="position", how="left")
print(df.head(20))


# Create output directory if it doesn't exist
output_dir = Path("../output/1_stitched")
output_dir.mkdir(parents=True, exist_ok=True)

for well in df['well'].unique():
    well_df = df[df['well'] == well]
    
    for channel in well_df['channel'].unique():
        channel_df = well_df[well_df['channel'] == channel]
        
        # Create a blank canvas for 2x2 grid (assuming all images have the same dimensions)
        # Read first image to get dimensions
        first_img_path = channel_df.iloc[0]['file_path']
        first_img = cv2.imread(str(first_img_path))
        h, w = first_img.shape[:2]
        
        # Calculate effective dimensions considering 8% overlap
        effective_w = int(w * 0.92)  # 92% of width (accounting for 8% overlap)
        effective_h = int(h * 0.92)  # 92% of height (accounting for 8% overlap)
        
        # Create a canvas with appropriate size for 2x2 grid with overlap
        total_width = w + effective_w
        total_height = h + effective_h
        grid_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # Place each image in the grid based on well_position
        for _, row in channel_df.iterrows():
            
            img = cv2.imread(str(row['file_path']))
            
            # well_position values are 0, 1, 2, 3 corresponding to positions in a 2x2 grid
            # 0: top-right, 1: bottom-right, 2: top-left, 3: bottom-left
            col = 1 - (row['well_position'] // 2)  # 1 for right (0,1), 0 for left (2,3)
            row_idx = row['well_position'] % 2  # 0 for top (0,2), 1 for bottom (1,3)
            
            # Calculate starting position with overlap
            start_x = col * effective_w
            start_y = row_idx * effective_h
            
            # Place the image in the grid
            grid_img[start_y:start_y+h, start_x:start_x+w] = img
        
        # Save the stitched image
        output_filename = f"well_{well}_channel_{channel}.png"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), grid_img)

print(f"Saved stitched images to {output_dir}")

