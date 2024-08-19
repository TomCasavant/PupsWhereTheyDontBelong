import torchvision
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from dog import Dog
from streetview import Streetview
from subclub import Subclub
import toml

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

dog = Dog()
config = toml.load("config.toml")

api_key = config["Streetview"]["api_key"]
streetview = Streetview(api_key)
streetview_image = streetview.fetch_random_image_from_mapillary()
dog_image = dog.get_cutout()

subclub = Subclub(config["Subclub"]["api_key"])

if dog_image is not None:
    result_image = paste_dog(dog_image, streetview_image.image)
    result_image.save("output_image.png")
    result_image.show()

    with open("firstnames.txt") as f:
        firstnames = f.read().splitlines()
    with open("secondnames.txt") as f:
        secondnames = f.read().splitlines()

    name = f"{np.random.choice(firstnames)} {np.random.choice(secondnames)}"

    print(f"Generated Name: {name}")
    print(f"Street View image saved to: {streetview_image.image_path}")
    print(f"Country: {streetview_image.country}")
    print(f"Latitude: {streetview_image.lat}, Longitude: {streetview_image.lon}")
    # http://www.openstreetmap.org/?mlat=latitude&mlon=longitude&zoom=12
    print(f"OpenStreetMap URL: http://www.openstreetmap.org/?mlat={streetview_image.lat}&mlon={streetview_image.lon}&zoom=12")

    with open("phrases.txt") as f:
        phrases = f.read().splitlines()
    phrase = np.random.choice(phrases)
    # Phrases look like What is {name} doing in {country}?
    phrase = phrase.replace("{name}", name)
    phrase = phrase.replace("{country}", streetview_image.country)
    print(f"Generated Phrase: {phrase}")
else:
    print("No dog detected in the source image.")
