import os
from PIL import Image
from logger import log_message
from config import BASE_ALPHA_MIN, BASE_ALPHA_MAX

def get_image_dimensions(directory_path: str):
    """
    Get dimensions of all images in the specified directory.
    """
    image_files = [f for f in os.listdir(directory_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                   and os.path.isfile(os.path.join(directory_path, f))]

    if not image_files:
        raise FileNotFoundError(f"No valid image files found in directory: {directory_path}")

    return [Image.open(os.path.join(directory_path, image_file)).size
            for image_file in image_files]

def adjust_alpha_range(picture: tuple, bounding_box_area: float, base_alpha_range: tuple = (BASE_ALPHA_MIN, BASE_ALPHA_MAX)):
    """
    Adjust the alpha range for an image based on its size relative to the bounding box.
    """
    picture_area = picture[0] * picture[1]
    area_ratio = picture_area / bounding_box_area
    adjusted_alpha_min = base_alpha_range[0] + 10 * ((area_ratio > 0.8) - (area_ratio < 0.2))
    
    if adjusted_alpha_min < base_alpha_range[0]:
        log_message("Adjusted Alpha Below Minimum for Image Size", f"{picture[0]}x{picture[1]}")
        log_message("Area Ratio", f"{area_ratio:.2f}")
        log_message("Adjusted Alpha Range", f"({adjusted_alpha_min}, {base_alpha_range[1]})")
        log_message("Bounding Box Area", f"{bounding_box_area}")

    return adjusted_alpha_min, base_alpha_range[1]
