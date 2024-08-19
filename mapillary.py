import os
import requests
from PIL import Image
from io import BytesIO
import time

def reverse_geocode(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
    headers = {
        'User-Agent': 'PupsWhereTheyDontBelong/1.0 (your.email@example.com)'
    }

    for attempt in range(2):
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                components = [
                    address.get('city', address.get('town', '')),
                    address.get('state', ''),
                    address.get('country', '')
                ]

                # Remove empty components and join with commas
                location_name = ', '.join(filter(None, components))
                return location_name if location_name else None

        except requests.RequestException:
            time.sleep(3)  # Wait for 3 seconds before retrying
            pass

    # Both attempts failed - return None
    return None

class MapillaryImage:
    def __init__(self, image, image_path, image_url, lat, lon, creator):
        self.image = image
        self.image_path = image_path
        self.lat = lat
        self.lon = lon
        self.creator = creator
        self.image_url = image_url

    @property
    def attribution(self):
        return f"<a href='{self.image_url}'>Photo</a> by <a href='https://www.mapillary.com/app/user/{self.creator}'>{self.creator}</a>"

    @property
    def location(self):
        return reverse_geocode(self.lat, self.lon)



class Mapillary:
    API_KEY: str = None
    OUTPUT_DIR: str = "images"
    MAPILLARY_IMAGE_ENDPOINT: str = "https://graph.mapillary.com/images"

    def __init__(self, api_key):
        self.API_KEY = api_key

    def find_latest_image(self):
        image_files = []
        for root, dirs, files in os.walk(self.OUTPUT_DIR):
            for file in files:
                if file.endswith(".jpg"):
                    image_files.append(os.path.join(root, file))

        if image_files:
            return max(image_files, key=os.path.getmtime)

        return None  # No image files found

    def fetch_random_image_from_mapillary(self):
        headers = {
            'Authorization': f'Bearer {self.API_KEY}'
        }

        params = {
            'fields': 'id,computed_geometry,creator.username,thumb_1024_url',
            'limit': 1,
            'bbox': '[-180,-90,180,90]'  # Bounding box covering the entire globe
        }

        response = requests.get(self.MAPILLARY_IMAGE_ENDPOINT, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                image_data = data['data'][0]

                image_id = image_data['id']
                lon, lat = image_data['computed_geometry']['coordinates']

                # Download the image
                image_url = image_data['thumb_1024_url']
                image_response = requests.get(image_url)

                if image_response.status_code == 200:
                    image = Image.open(BytesIO(image_response.content))

                    # Save image locally
                    if not os.path.exists(self.OUTPUT_DIR):
                        os.makedirs(self.OUTPUT_DIR)
                    image_path = os.path.join(self.OUTPUT_DIR, f"{image_id}.jpg")
                    image.save(image_path)

                    # Extract attribution details
                    creator = image_data['creator']['username']
                    # Attribution = <a href='image link'>Image Title</a> by <a href='author profile'>Author</a>


                    return MapillaryImage(image, image_path, image_url, lat, lon, creator)

        return None  # No images found
