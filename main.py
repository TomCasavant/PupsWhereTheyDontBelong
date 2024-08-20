import os
import numpy as np
import toml
from dog import Dog
from mapillary import Mapillary
from subclub import Subclub

# This script is run via cron, so we need to set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class DogMapillaryPost:
    """
    A class to create and post content by combining random dog images from the Dog CEO API
    and random photos from Mapillary.
    """

    def __init__(self, config_path="config.toml"):
        """
        Initialize the DogMapillaryPost with the configuration from the specified TOML file.

        Args:
            config_path (str): The path to the configuration TOML file.
        """
        self.config = toml.load(config_path)
        self.dog = Dog()
        self.mapillary = Mapillary(self.config["Mapillary"]["api_key"])
        self.subclub = Subclub(self.config["Subclub"]["api_key"])

    def generate_name(self):
        """
        Generate a random name for the dog by combining a random first name and last name.

        Returns:
            str: The generated dog name.
        """
        try:
            with open("assets/firstnames.txt") as f:
                firstnames = f.read().splitlines()
            with open("assets/secondnames.txt") as f:
                secondnames = f.read().splitlines()
            name = f"{np.random.choice(firstnames)} {np.random.choice(secondnames)}"
            return name
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return "Unknown Name"

    def generate_phrase(self, name, location):
        """
        Generate a phrase by selecting a random line from a phrases file and filling in placeholders.

        Args:
            name (str): The generated name of the dog.
            location (str): The location associated with the Mapillary image.

        Returns:
            str: The generated phrase with placeholders replaced.
        """
        try:
            phrase_file = "assets/phrases.txt" if location else "assets/phrases_unknown_location.txt"
            with open(phrase_file) as f:
                phrases = f.read().splitlines()
            phrase = np.random.choice(phrases)
            phrase = phrase.replace("{name}", name)
            phrase = phrase.replace("{country}", location or "Unknown Location")
            return phrase
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return "No phrase available."

    def create_post_content(self, phrase, mapillary_image):
        """
        Create the content of the post, including the generated phrase and attributions.

        Args:
            phrase (str): The generated phrase for the post.
            mapillary_image (MapillaryImage): The Mapillary image object with location data.

        Returns:
            str: The formatted content for the post.
        """
        if mapillary_image.location:
            osm_url = f"http://www.openstreetmap.org/?mlat={mapillary_image.lat}&mlon={mapillary_image.lon}&zoom=12"
            location_info = f"<br /><a href='{osm_url}'>{mapillary_image.location}</a>"
        else:
            location_info = ""

        result = f"{phrase}<br /><br />{mapillary_image.attribution}{location_info}"
        result += f"<br />{self.dog.attribution}"
        return result

    def run(self):
        """
        Execute the main logic of fetching a Mapillary image and a dog cutout,
        combining them, and posting the result.
        """
        mapillary_image = self.mapillary.fetch_random_image_from_mapillary()
        dog_image = self.dog.get_cutout()

        if dog_image is not None:
            result_image = self.dog.paste_to_image(dog_image, mapillary_image.image)
            result_image.save("output_image.png")
            result_image.show()

            name = self.generate_name()
            print(f"Generated Name: {name}")
            print(f"Mapillary image saved to: {mapillary_image.image_path}")
            print(f"Country: {mapillary_image.location}")
            print(f"Latitude: {mapillary_image.lat}, Longitude: {mapillary_image.lon}")

            phrase = self.generate_phrase(name, mapillary_image.location)
            print(f"Generated Phrase: {phrase}")

            post_content = self.create_post_content(phrase, mapillary_image)
            response = self.subclub.post(post_content)
        else:
            print("No dog detected in the source image.")


if __name__ == "__main__":
    dog_street_view_post = DogMapillaryPost()
    dog_street_view_post.run()

