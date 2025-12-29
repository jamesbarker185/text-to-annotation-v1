import requests
from PIL import Image
import io

# Create a dummy image
img = Image.new('RGB', (100, 100), color='red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

files = {'file': ('test.png', img_byte_arr, 'image/png')}
data = {'prompts': 'cat, dog'}

try:
    print("Sending request...")
    response = requests.post('http://localhost:8000/api/detect', files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
