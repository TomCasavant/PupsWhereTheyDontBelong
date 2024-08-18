import os
import subprocess
import pycountry
from PIL import Image


class StreetviewImage:
    image: Image
    image_path: str
    country: str
    lat: str
    lon: str

    def __init__(self, image, image_path, country, lat, lon):
        self.image = image
        self.image_path = image_path
        self.country = country
        self.lat = lat
        self.lon = lon


class Streetview:

    API_KEY: str = None
    OUTPUT_DIR: str = "images"

    def __init__(self, api_key,):
        self.API_KEY = api_key


    def find_latest_image(self):
        # TODO: I can probably use glob to do this faster and more efficiently
        image_files = []
        for root, dirs, files in os.walk(self.OUTPUT_DIR):
            for file in files:
                if file.endswith(".jpg"):
                    image_files.append(os.path.join(root, file))

        if image_files:
            return max(image_files, key=os.path.getmtime)

        return None # No images files found, error in street-view-randomizer?

    '''
        Uses the street-view-randomizer CLI tool to generate a random street view image.
    '''
    # TODO: Is there a better way to extract random streetview coords? I tried using random coordinates but they never lined up with a streetview address
    def generate_random_streetview_image(self):
        # Ensure the output directory exists
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

        # Generate the command
        command = f"street-view-randomizer --api-key={self.API_KEY} -o {self.OUTPUT_DIR}"

        # Run the command
        subprocess.run(command, shell=True)
        latest_image_path = self.find_latest_image()
        if latest_image_path:
            # Extract the country code from the directory name
            country_code = os.path.basename(os.path.dirname(latest_image_path))

            # Convert the country code to the actual country name
            country = pycountry.countries.get(alpha_3=country_code.upper())
            country_name = country.name if country else None

            # Extract the filename and parse lat/lon
            filename = os.path.basename(latest_image_path)
            parts = filename.split("_")

            lon = parts[0]
            lat = parts[1]

            return StreetviewImage(Image.open(latest_image_path), latest_image_path, country_name, lat, lon)

        return None # No images found, error in street-view-randomizer?
