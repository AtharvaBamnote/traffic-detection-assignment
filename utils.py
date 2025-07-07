import cv2
import numpy as np
import requests

def download_image(url):
    try:
        response = requests.get(url)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None
