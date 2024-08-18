import torchvision
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from dog import Dog
from streetview import Streetview

def paste_dog(dog_image, target_image, x_offset=50, y_offset=50):
    dog_image_rgba = dog_image.convert("RGBA")
    target_image_rgba = target_image.convert("RGBA")

    dog_image_cv = np.array(dog_image_rgba)
    target_image_cv = np.array(target_image_rgba)

    h, w, _ = dog_image_cv.shape
    target_h, target_w, _ = target_image_cv.shape

    if x_offset + w > target_w or y_offset + h > target_h:
        max_width = target_w - x_offset
        max_height = target_h - y_offset
        scaling_factor = min(max_width / w, max_height / h)
        
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        dog_image_cv = cv2.resize(dog_image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        dog_image_rgba = Image.fromarray(dog_image_cv, "RGBA")

        # Update dimensions after resizing
        h, w, _ = dog_image_cv.shape

    # Create an empty RGBA image for pasting
    result_image_cv = target_image_cv.copy()
    
    # Create a mask from the alpha channel of the dog image
    mask = dog_image_cv[..., 3]
    mask = cv2.merge((mask, mask, mask))  # Convert mask to 3 channels
    
    # Paste the dog image onto the target image
    for c in range(3):  # For each color channel
        result_image_cv[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
            (dog_image_cv[..., 3] / 255.0 * dog_image_cv[..., c] +
             (1 - dog_image_cv[..., 3] / 255.0) * target_image_cv[y_offset:y_offset+h, x_offset:x_offset+w, c])
    
    # Apply the alpha channel to the result image
    alpha_channel = np.maximum(mask[..., 0], result_image_cv[y_offset:y_offset+h, x_offset:x_offset+w, 3])
    result_image_cv[y_offset:y_offset+h, x_offset:x_offset+w, 3] = alpha_channel

    result_image = Image.fromarray(result_image_cv, "RGBA")
    return result_image

def fetch_random_dog_image():
    # Fetch a random dog image URL from the Dog CEO API
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    data = response.json()
    image_url = data["message"]
    
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    return image

dog = Dog()
api_key=""
streetview = Streetview(api_key)
streetview_image = streetview.generate_random_streetview_image()
dog_image = dog.get_cutout()
if dog_image is not None:
    result_image = paste_dog(dog_image, streetview_image.image)
    result_image.save("output_image.png")
    result_image.show()
    print(f"Street View image saved to: {streetview_image.image_path}")
    print(f"Country: {streetview_image.country}")
    print(f"Latitude: {streetview_image.lat}, Longitude: {streetview_image.lon}")
else:
    print("No dog detected in the source image.")
