import cv2
import numpy as np

def preprocess_image(image_path, target_height=480):
    """Loads image and downgrades it to 480p height."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_img

def contains_person(image):
    """Checks if a person is in the image using Haar Cascades."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return len(faces) > 0

def get_pixel_mapping(source_img, target_img_path):
    """
    Calculates the start (x,y) and end (x,y) for every pixel.
    Returns: start_coords, end_coords, colors
    """
    # 1. Prepare Target (Modi)
    target = cv2.imread(target_img_path)
    # Resize target to match source EXACTLY
    target = cv2.resize(target, (source_img.shape[1], source_img.shape[0]))

    # 2. Flatten images
    # We work with flat arrays to sort pixels easily
    source_flat = source_img.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)
    
    num_pixels = source_flat.shape[0]

    # 3. Calculate Luminance for sorting
    # (How bright is the pixel?)
    def get_lum(pixels):
        return np.dot(pixels, [0.114, 0.587, 0.299])

    print("  > Sorting pixels (solving assignment problem)...")
    source_lum = get_lum(source_flat)
    target_lum = get_lum(target_flat)

    # 4. Get the sorting indices
    # source_indices[0] is the index of the darkest pixel in source
    source_indices = np.argsort(source_lum)
    target_indices = np.argsort(target_lum)

    # 5. Create Coordinate Arrays
    # We want to move the i-th darkest source pixel to the position of the i-th darkest target pixel
    
    # Generate grid of coordinates (y, x)
    h, w = source_img.shape[:2]
    rows, cols = np.indices((h, w))
    
    # Flatten coordinates
    flat_coords = np.column_stack((cols.ravel(), rows.ravel())) # (x, y)
    
    # Start positions: The actual location of the sorted source pixels
    start_coords = flat_coords[source_indices]
    
    # End positions: The location where they MUST go (the sorted target spots)
    end_coords = flat_coords[target_indices]
    
    # Colors: The actual colors of the source pixels we are moving
    colors = source_flat[source_indices]

    return start_coords, end_coords, colors