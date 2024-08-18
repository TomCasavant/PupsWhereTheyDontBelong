import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_dog_mask(image, threshold=0.5):
    # Ensure image is in RGB format
    image = image.convert("RGB")
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    masks = predictions[0]['masks']
    labels = predictions[0]['labels']

    dog_mask = None
    for i, label in enumerate(labels):
        if label == 18:  # 18 is the class index for "dog" in COCO dataset
            dog_mask = masks[i, 0].mul(255).byte().cpu().numpy()

            # Apply thresholding to make the mask tighter
            _, dog_mask = cv2.threshold(dog_mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            dog_mask = cv2.erode(dog_mask, kernel, iterations=1)
            dog_mask = cv2.dilate(dog_mask, kernel, iterations=1)
            
            # Find contours and create a new mask that closely follows the dog shape
            contours, _ = cv2.findContours(dog_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            new_mask = np.zeros_like(dog_mask)
            cv2.drawContours(new_mask, contours, -1, 255, thickness=cv2.FILLED)

            dog_mask = new_mask
            break
    
    return dog_mask

def extract_dog(image, dog_mask):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create an alpha channel with the same size as the image
    alpha_channel = np.ones((image_cv.shape[0], image_cv.shape[1]), dtype=np.uint8) * 255
    
    # Apply the mask to the alpha channel
    alpha_channel[dog_mask == 0] = 0
    
    # Add the alpha channel to the image
    bgr_image = cv2.merge((image_cv[..., 0], image_cv[..., 1], image_cv[..., 2], alpha_channel))
    
    # Convert to PIL Image format with transparency
    dog_image_pil = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2RGBA))
    return dog_image_pil

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

source_image = fetch_random_dog_image()
target_image_path = "streetview.png"

target_image = Image.open(target_image_path).convert("RGBA")

dog_mask = get_dog_mask(source_image, threshold=0.35) 

if dog_mask is not None:
    dog_image = extract_dog(source_image, dog_mask)
    result_image = paste_dog(dog_image, target_image)
    result_image.save("output_image.png")
    result_image.show()
else:
    print("No dog detected in the source image.")
