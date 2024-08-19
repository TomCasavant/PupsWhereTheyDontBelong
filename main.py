import numpy as np
import toml
from dog import Dog
from mapillary import Mapillary
from subclub import Subclub

# This script is run via cron so we need to set the working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DogMapillaryPost:
    def __init__(self, config_path="config.toml"):
        self.config = toml.load(config_path)
        self.dog = Dog()
        self.mapillary = Mapillary(self.config["Mapillary"]["api_key"])
        self.subclub = Subclub(self.config["Subclub"]["api_key"])

    '''
        Combines a random first name and last name to create Dog's new name
    '''
    def generate_name(self):
        with open("assets/firstnames.txt") as f:
            firstnames = f.read().splitlines()
        with open("assets/secondnames.txt") as f:
            secondnames = f.read().splitlines()
        name = f"{np.random.choice(firstnames)} {np.random.choice(secondnames)}"
        return name

    '''
        Generates a phrase from the phrases.txt file, fills in {country} and {name} placeholders
    '''
    def generate_phrase(self, name, location):
        phrase_file = "assets/phrases.txt"
        if not location:
            phrase_file = "assets/phrases_unknown_location.txt"

        with open(phrase_file) as f:
            phrases = f.read().splitlines()
        phrase = np.random.choice(phrases)
        phrase = phrase.replace("{name}", name)
        phrase = phrase.replace("{country}", location)
        return phrase

    '''
        Create the post content with the generated phrase and attribution
    '''
    def create_post_content(self, phrase, mapillary_image):
        if mapillary_image.location:
            osm_url = f"http://www.openstreetmap.org/?mlat={mapillary_image.lat}&mlon={mapillary_image.lon}&zoom=12"
            location_info = f"<br /><a href='{osm_url}'>{mapillary_image.location}</a>"
        else:
            location_info = ""

        result = f"{phrase}<br /><br />{mapillary_image.attribution}{location_info}"
        result += f"<br />{self.dog.attribution}"
        return result

    def run(self):
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
