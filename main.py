import processor
import animator
import os

# --- CONFIGURATION ---
INPUT_FILE = "assets/input.jpg"
TARGET_FILE = "assets/Prime_Minister_of_India_Narendra_Modi.jpg" # Make sure this exists!
OUTPUT_VIDEO = "modi_transformation.mp4"
# ---------------------

def main():
    print("--- Starting Project Modi-fication ---")
    
    # 1. Check if files exist
    if not os.path.exists(INPUT_FILE) or not os.path.exists(TARGET_FILE):
        print("❌ Error: Please put 'input.jpg' and 'modi.jpg' in the 'assets' folder.")
        return

    # 2. Preprocessing (Downgrade to 480p)
    print("1. Preprocessing input...")
    source_img = processor.preprocess_image(INPUT_FILE)
    
    # 3. Person Detection
    print("2. Scanning for person...")
    if not processor.contains_person(source_img):
        print("❌ Security Check Failed: No person detected in input image.")
        return
    print("✅ Person detected.")

    # 4. Math (Calculate Pixel Paths)
    print("3. Calculating pixel paths...")
    starts, ends, colors = processor.get_pixel_mapping(source_img, TARGET_FILE)

    # 5. Animation
    print("4. Rendering video...")
    animator.create_animation(
        starts, 
        ends, 
        colors, 
        source_img.shape, 
        output_file=OUTPUT_VIDEO
    )

if __name__ == "__main__":
    main()