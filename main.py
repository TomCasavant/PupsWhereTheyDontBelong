import numpy as np
import toml
from dog import Dog
from streetview import Streetview
from subclub import Subclub

class DogStreetViewPost:
    def __init__(self, config_path="config.toml"):
        self.config = toml.load(config_path)
        self.dog = Dog()
        self.streetview = Streetview(self.config["Mapillary"]["api_key"])
        self.subclub = Subclub(self.config["Subclub"]["api_key"])

    def generate_name(self):
        with open("firstnames.txt") as f:
            firstnames = f.read().splitlines()
        with open("secondnames.txt") as f:
            secondnames = f.read().splitlines()
        name = f"{np.random.choice(firstnames)} {np.random.choice(secondnames)}"
        return name

    def generate_phrase(self, name, location):
        with open("phrases.txt") as f:
            phrases = f.read().splitlines()
        phrase = np.random.choice(phrases)
        phrase = phrase.replace("{name}", name)
        phrase = phrase.replace("{country}", location or "an unknown location")
        return phrase

    def create_post_content(self, phrase, streetview_image):
        if streetview_image.location:
            osm_url = f"http://www.openstreetmap.org/?mlat={streetview_image.lat}&mlon={streetview_image.lon}&zoom=12"
            location_info = f"<br /><a href='{osm_url}'>{streetview_image.location}</a>"
        else:
            location_info = ""

        result = f"{phrase}<br /><br />{streetview_image.attribution}{location_info}"
        result += f"<br />{self.dog.attribution}"
        return result

    def run(self):
        streetview_image = self.streetview.fetch_random_image_from_mapillary()
        dog_image = self.dog.get_cutout()

        if dog_image is not None:
            result_image = self.dog.paste_to_image(dog_image, streetview_image.image)
            result_image.save("output_image.png")
            result_image.show()

            name = self.generate_name()
            print(f"Generated Name: {name}")
            print(f"Street View image saved to: {streetview_image.image_path}")
            print(f"Country: {streetview_image.location}")
            print(f"Latitude: {streetview_image.lat}, Longitude: {streetview_image.lon}")

            phrase = self.generate_phrase(name, streetview_image.location)
            print(f"Generated Phrase: {phrase}")

            post_content = self.create_post_content(phrase, streetview_image)
            response = self.subclub.post(post_content)
        else:
            print("No dog detected in the source image.")


if __name__ == "__main__":
    dog_street_view_post = DogStreetViewPost()
    dog_street_view_post.run()
