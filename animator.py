import cv2
import numpy as np
from tqdm import tqdm # Progress bar

def create_animation(start_coords, end_coords, colors, output_shape, output_file="output.mp4", frames=120):
    """
    Generates the video file interpolating from start to end.
    """
    h, w = output_shape[:2]
    
    # Video Writer Setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (w, h))
    
    print(f"  > Generating {frames} frames of animation...")

    # We use float32 for smooth calculation, int for final pixel placement
    start_coords = start_coords.astype(np.float32)
    end_coords = end_coords.astype(np.float32)
    
    # Animation Loop
    for t in tqdm(np.linspace(0, 1, frames)):
        # Linear Interpolation: Current = Start * (1-t) + End * t
        # Adding some noise 'np.random' can make it look more like "fluid" or "particles"
        current_pos = (start_coords * (1 - t)) + (end_coords * t)
        
        # Add a little sine wave wobble for "fluid" feel
        wobble = np.sin(t * np.pi) * (np.random.normal(0, 5, current_pos.shape))
        current_pos += wobble

        # Prepare blank frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Clip coordinates to stay inside image bounds
        cur_x = np.clip(current_pos[:, 0], 0, w - 1).astype(int)
        cur_y = np.clip(current_pos[:, 1], 0, h - 1).astype(int)
        
        # Place pixels
        # Note: This overwrites if multiple pixels land on same spot (simple collision)
        frame[cur_y, cur_x] = colors
        
        out.write(frame)

    out.release()
    print(f"✅ Animation saved to {output_file}")