import streamlit as st
import cv2
import numpy as np
import processor
import tempfile
import os

# --- CONFIGURATION ---
TARGET_PATH = "assets/Prime_Minister_of_India_Narendra_Modi.jpg"
TOTAL_FRAMES = 150  # Fixed duration: 5 seconds @ 30fps
# ---------------------

st.set_page_config(page_title="The Modi Effect", layout="centered")

st.title("🇮🇳 The Modi-fication Project")
st.markdown("Upload your photo. The AI will generate a video transformation.")

# --- HELPER FUNCTION ---
def generate_video_file(processed_img, starts, ends, colors):
    """
    Generates a .webm file and returns the path.
    """
    h, w = processed_img.shape[:2]
    
    # Create a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    output_path = tfile.name
    
    # VP09 codec for web compatibility
    fourcc = cv2.VideoWriter_fourcc(*'vp09') 
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    # Convert coords to float for calculation
    starts = starts.astype(np.float32)
    ends = ends.astype(np.float32)

    # Progress bar in the UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, t in enumerate(np.linspace(0, 1, TOTAL_FRAMES)):
        # Interpolate
        current_pos = (starts * (1 - t)) + (ends * t)
        
        # Add fluid noise
        noise_amount = 5.0
        wobble = np.sin(t * np.pi) * (np.random.normal(0, noise_amount, current_pos.shape))
        current_pos += wobble
        
        # Build frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cur_x = np.clip(current_pos[:, 0], 0, w - 1).astype(int)
        cur_y = np.clip(current_pos[:, 1], 0, h - 1).astype(int)
        frame[cur_y, cur_x] = colors
        
        out.write(frame)
        
        # Update UI progress every 10 frames
        if i % 10 == 0:
            progress = (i + 1) / TOTAL_FRAMES
            progress_bar.progress(progress)
            status_text.text(f"Rendering frame {i+1}/{TOTAL_FRAMES}...")

    out.release()
    progress_bar.empty() # Remove bar when done
    status_text.empty()
    return output_path

# --- MAIN APP ---

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, 1)
    
    # Display ONLY the input image
    st.subheader("Preview")
    st.image(input_img, channels="BGR", width=400)

    if st.button("🎬 Generate Video", type="primary"):
        with st.spinner("Calculating pixel trajectories..."):
            # 1. Preprocess
            temp_path = "temp_input.jpg"
            cv2.imwrite(temp_path, input_img)
            processed_img = processor.preprocess_image(temp_path)
            
            # 2. Check Person
            if not processor.contains_person(processed_img):
                st.error("❌ No person detected. Try a different photo.")
            else:
                # 3. Calculate Math
                try:
                    starts, ends, colors = processor.get_pixel_mapping(processed_img, TARGET_PATH)
                    
                    # 4. Generate Video File
                    video_path = generate_video_file(processed_img, starts, ends, colors)
                    
                    # 5. Display Player
                    st.success("Rendering Complete!")
                    st.video(video_path)
                    
                except Exception as e:
                    st.error(f"Error: {e}")