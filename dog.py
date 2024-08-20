import torch
import torchvision
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


class Dog:
    """
    A class for working with dog images.
    """

    API_URL: str = "https://dog.ceo/api/breeds/image/random"
    DOG_INDEX: int = 18  # 18 is the class index for "dog" in COCO dataset
    model = None
    image_url: str = None

    def __init__(self):
        """Initialize the Dog class, download model."""
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.eval()

    def fetch_random_dog_image(self) -> Image:
        """
        Fetch a random dog image from the Dog CEO API.

        Returns:
            Image: AnImage object of the fetched dog.
        """
        response = requests.get(self.API_URL, timeout=10)  # Added timeout here
        data = response.json()
        self.image_url = data["message"]

        response = requests.get(self.image_url, timeout=10)  # Added timeout here
        image = Image.open(BytesIO(response.content))

        return image

    @property
    def attribution(self):
        """Provide the attribution for the dog image source."""
        return f"<a href='{self.image_url}'>Dog Source</a>"

    def get_dog_mask(self, image: Image, threshold=0.5) -> np.ndarray:
        """
        Generate a mask for the dog in the provided image.

        Args:
            image (Image): The image containing the dog.
            threshold (float): The threshold for the mask.
        """
        # Convert image to tensor
        transform = torchvision.transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Filter for dog label and create mask
        for i, label in enumerate(predictions[0]["labels"]):
            if label.item() == self.DOG_INDEX:
                mask = predictions[0]["masks"][i, 0].cpu().numpy()
                binary_mask = (mask > threshold).astype(np.uint8) * 255
                return binary_mask

        return None

    def extract_dog(self, image: Image, dog_mask: np.ndarray) -> Image:
        """
        Extract the dog from the image using the provided mask.

        Args:
            image (Image): The original image.
            dog_mask (np.ndarray): The mask for the dog.
        """
        image_np = np.array(image)
        alpha_channel = np.where(dog_mask > 0, 255, 0).astype(np.uint8)
        image_with_alpha = np.dstack((image_np, alpha_channel))

        dog_image_pil = Image.fromarray(image_with_alpha)
        return dog_image_pil

    def get_cutout(self) -> Image:
        """
        Get a cutout of the dog from a random image.

        Returns:
            Image: An Image object of the dog cutout with transparency, or None if failed.
        """
        image = self.fetch_random_dog_image()

        if image is None:
            return None

        dog_mask = self.get_dog_mask(image)

        if dog_mask is None:
            return None

        dog_cutout = self.extract_dog(image, dog_mask)

        return dog_cutout

    def paste_to_image(self, dog_image, target_image, x_offset=50, y_offset=50):
        """
        Paste the dog cutout onto a target image.

        Args:
            dog_image (Image): The dog cutout image.
            target_image (Image): The image to paste the dog onto.
            x_offset (int): The x-coordinate of the top-left corner to start pasting.
            y_offset (int): The y-coordinate of the top-left corner to start pasting.

        Returns:
            Image: The target image with the dog cutout pasted onto it.
        """
        #TODO: Experiment with offset values to more randomly place the dog
        dog_image = dog_image.convert("RGBA")
        target_image = target_image.convert("RGBA")

        # Resize the dog image if necessary
        target_width, target_height = target_image.size
        dog_width, dog_height = dog_image.size

        if x_offset + dog_width > target_width or y_offset + dog_height > target_height:
            max_width = target_width - x_offset
            max_height = target_height - y_offset
            scaling_factor = min(max_width / dog_width, max_height / dog_height)
            new_size = (
                int(dog_width * scaling_factor),
                int(dog_height * scaling_factor),
            )
            dog_image = dog_image.resize(new_size, Image.Resampling.LANCZOS)

        # Paste the dog image onto the target image with transparency
        target_image.paste(dog_image, (x_offset, y_offset), dog_image)

        return target_image
