import torch
import torchvision
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
# import the weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

class Dog:

    API_URL: str = "https://dog.ceo/api/breeds/image/random"
    model = None
    image_url = None

    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()

    '''
    Fetch a random dog image from the Dog CEO API
    '''
    def fetch_random_dog_image(self) -> Image:
        # Fetch a random dog image URL from the Dog CEO API
        response = requests.get(self.API_URL)
        data = response.json()
        self.image_url = data["message"]

        response = requests.get(self.image_url)
        image = Image.open(BytesIO(response.content))

        return image

    @property
    def attribution(self):
        return f"<a href='{self.image_url}'>Dog Source</a>"

    '''
    Get the mask of the dog in the image
    '''
    def get_dog_mask(self, image, threshold=0.5) -> np.ndarray:
        image = image.convert("RGB")

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(image_tensor)

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

    '''
    Extract the dog from the image using the mask
    '''
    def extract_dog(self, image, dog_mask) -> Image:
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

    '''
    Get a cutout of a dog image
    '''
    def get_cutout(self) -> Image:
        # Fetch a random dog image
        image = self.fetch_random_dog_image()

        if image is None:
            # Image fetch failed
            return None

        # Get the dog mask
        dog_mask = self.get_dog_mask(image)

        if dog_mask is None:
            # Dog could not be found in image
            return None

        # Extract the dog from the image using the mask
        dog_cutout = self.extract_dog(image, dog_mask)

        return dog_cutout

    '''
        Copies the dog cutout to the new image
    '''
    def paste_to_image(self, dog_image, target_image, x_offset=50, y_offset=50):
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

            # Update dimensions after resizing
            h, w, _ = dog_image_cv.shape

        # Create an empty RGBA image for pasting
        result_image_cv = target_image_cv.copy()

        # Create a mask from the alpha channel of the dog image
        mask = dog_image_cv[..., 3]
        mask = cv2.merge((mask, mask, mask))  # Convert mask to 3 channels

        # Paste the dog image onto the target image
        for c in range(3):  # For each color channel
            result_image_cv[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
                (dog_image_cv[..., 3] / 255.0 * dog_image_cv[..., c] +
                 (1 - dog_image_cv[..., 3] / 255.0) * target_image_cv[y_offset:y_offset + h, x_offset:x_offset + w, c])

        # Apply the alpha channel to the result image
        alpha_channel = np.maximum(mask[..., 0], result_image_cv[y_offset:y_offset + h, x_offset:x_offset + w, 3])
        result_image_cv[y_offset:y_offset + h, x_offset:x_offset + w, 3] = alpha_channel

        result_image = Image.fromarray(result_image_cv, "RGBA")
        return result_image