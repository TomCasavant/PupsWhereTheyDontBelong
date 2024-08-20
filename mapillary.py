import os
import time
import requests
from PIL import Image
from io import BytesIO


def reverse_geocode(lat, lon):
    """
    Perform a reverse geocoding lookup using the OpenStreetMap Nominatim API.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.

    Returns:
        str or None: A string representing the location (city, state, country) or None if not found.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    headers = {"User-Agent": "PupsWhereTheyDontBelong/1.0 (your.email@example.com)"}

    for _ in range(2):  # Retry twice
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})
                components = [
                    address.get("city", address.get("town", "")),
                    address.get("state", ""),
                    address.get("country", ""),
                ]
                location_name = ", ".join(filter(None, components))
                return location_name if location_name else None
        except requests.RequestException:
            time.sleep(3)  # Wait for 3 seconds before retrying

    return None  # Both attempts failed


class MapillaryImage:
    """
    Represents an image fetched from Mapillary with metadata.
    """

    def __init__(self, image, image_path, image_url, lat, lon, creator):
        """
        Initialize the MapillaryImage object.

        Args:
            image (PIL.Image): The image object.
            image_path (str): Local path where the image is saved.
            image_url (str): URL of the image.
            lat (float): Latitude of the image location.
            lon (float): Longitude of the image location.
            creator (str): Username of the image creator.
        """
        self.image = image
        self.image_path = image_path
        self.lat = lat
        self.lon = lon
        self.creator = creator
        self.image_url = image_url

    @property
    def attribution(self):
        """
        Generate attribution text with links to the image and the creator.

        Returns:
            str: HTML string with attribution.
        """
        return f"<a href='{self.image_url}'>Photo</a> by <a href='https://www.mapillary.com/app/user/{self.creator}'>{self.creator}</a>"

    @property
    def location(self):
        """
        Perform reverse geocoding to get the location name.

        Returns:
            str or None: The location name (city, state, country) or None if not found.
        """
        return reverse_geocode(self.lat, self.lon)


class Mapillary:
    """
    Handles interactions with the Mapillary API to fetch and manage images.
    """

    OUTPUT_DIR = "images"
    MAPILLARY_IMAGE_ENDPOINT = "https://graph.mapillary.com/images"

    def __init__(self, api_key):
        """
        Initialize the Mapillary object with the given API key.

        Args:
            api_key (str): The API key for authenticating with the Mapillary API.
        """
        self.api_key = api_key

    def find_latest_image(self):
        """
        Find the latest image saved in the output directory.

        Returns:
            str or None: The path to the latest image or None if no images are found.
        """
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.OUTPUT_DIR)
            for file in files if file.endswith(".jpg")
        ]
        return max(image_files, key=os.path.getmtime) if image_files else None

    def fetch_random_image_from_mapillary(self):
        """
        Fetch a random image from the Mapillary API and download it.

        Returns:
            MapillaryImage or None: The MapillaryImage object if an image is found, otherwise None.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "fields": "id,computed_geometry,creator.username,thumb_1024_url",
            "limit": 1,
            "bbox": "[-180,-90,180,90]",  # Bounding box covering the entire globe
        }

        try:
            response = requests.get(self.MAPILLARY_IMAGE_ENDPOINT, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    image_data = data[0]
                    image_id = image_data["id"]
                    lon, lat = image_data["computed_geometry"]["coordinates"]

                    # Download the image
                    image_url = image_data["thumb_1024_url"]
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image = Image.open(BytesIO(image_response.content))

                        # Save the image locally
                        if not os.path.exists(self.OUTPUT_DIR):
                            os.makedirs(self.OUTPUT_DIR)
                        image_path = os.path.join(self.OUTPUT_DIR, f"{image_id}.jpg")
                        image.save(image_path)

                        creator = image_data["creator"]["username"]
                        return MapillaryImage(image, image_path, image_url, lat, lon, creator)
        except requests.RequestException as e:
            print(f"Error fetching image from Mapillary: {e}")

        return None  # No images found or an error occurred
