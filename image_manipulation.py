from PIL import Image
import numpy as np

# load and convert image to RGB format
def get_image(filename: str) -> Image.Image:
    img = Image.open(filename)
    img = img.convert('RGB') # RGB 3-channel array
    return img

# crop image to largest square image and downgrade to target sizing for future use
def reshape_image(img: Image.Image, target_size: int = 800, img_dir: str = "images_generated") -> tuple[Image.Image, str]:
    width, height = img.size
    min_dim = min(width, height)

    # crop center to create square image
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_cropped = img.crop((left, top, right, bottom))

    # LANCZOS resampling provides high-quality downscaling to preserve edge detection
    img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)
    img_resized.save(img_dir + "processed_input.jpg")
    return img_resized, img_dir

# converting PIL image to RGB 3-channel array
def get_rgb_data(img: Image.Image) -> np.ndarray:
    return np.array(img)