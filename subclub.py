"""
Example Curl Request:
curl --location 'https://sub.club/api/public/post' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $API_KEY' \
--data '{ "content": "Hello world" }'
"""
import requests

class Subclub:
    """
    A class to interact with the sub.club API for posting content.

    Attributes:
        BASE_URL (str): The base URL for the sub.club API.
        API_KEY (str): The API key used for authenticating requests.
    """

    BASE_URL = "https://sub.club/api/public"

    def __init__(self, api_key):
        """
        Initialize the sub.club object with the provided API key.

        Args:
            api_key (str): The API key for authenticating with the Subclub API.
        """
        self.API_KEY = api_key

    def post(self, content: str, media_ids=None):
        """
        Post a message to the sub.club API.

        Args:
            content (str): The content to be posted.

        Returns:
            dict: The JSON response from the API.
        """
        url = f"{self.BASE_URL}/post"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}",
        }
        data = {"content": content}
        if media_ids:
            data["mediaIds"] = media_ids
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to post content: {e}")
            return None

    def upload_media(self, file_path: str):
        """
        Upload media to the sub.club API
        """
        #TODO: Add proper documentation
        url = f"{self.BASE_URL}/media"
        headers = {
            "Authorization": f"Bearer {self.API_KEY}"
        }
        files = {
            "file": open(file_path, "rb")
        }

        try:
            response = requests.post(url, headers=headers, files=files)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to upload media: {e}")
            return None
