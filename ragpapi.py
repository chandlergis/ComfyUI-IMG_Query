import os
import io
import json
import requests
import torch
import numpy as np
from io import BytesIO
from PIL import Image

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageRequestNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {"default": "美女 车 雪山", "multiline": True}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "bm25_n": ("INT", {"default": 500, "min": 1, "max": 1000}),
                "retrieve_n": ("INT", {"default": 10, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fetch_image"
    CATEGORY = "🔥Image Request"

    def fetch_image(self, query, n, bm25_n, retrieve_n):
        url = "http://10.0.100.224:58000/rag_proxy/vl_result_fix"
        payload = json.dumps({
            "query": query,
            "n": n,
            "bm25_n": bm25_n,
            "retrieve_n": retrieve_n
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            json_data = response.json()
            if json_data['code'] == 200 and 'data' in json_data and len(json_data['data']) > 0:
                image_url = json_data['data'][0]['url']
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = Image.open(BytesIO(image_response.content))
                    output_t = pil2tensor(image_data)
                    return (output_t,)
                else:
                    raise Exception(f"Failed to fetch image from URL: {image_response.status_code}")
            else:
                raise Exception(f"API response error: {json_data.get('message', 'Unknown error')}")
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

NODE_CLASS_MAPPINGS = {
    "ImageRequestNode": ImageRequestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageRequestNode": "🔥Image Request Node",
}

