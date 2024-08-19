import requests

'''
Example Curl Request:
curl --location 'https://sub.club/api/public/post' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $API_KEY' \
--data '{ "content": "Hello world" }'
'''

class Subclub:

    BASE_URL: str = "https://sub.club/api/public"
    API_KEY: str = None

    def __init__(self, api_key):
        self.API_KEY = api_key

    """
    Post a message to the sub.club API
    Endpoint: /post
    """
    def post(self, content: str):
        url = f"{self.BASE_URL}/post"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        data = {
            "content": content
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()