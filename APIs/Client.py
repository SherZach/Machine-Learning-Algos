import requests
import json
from PIL import Image
import base64
from io import BytesIO

# r = requests.get('http://localhost:4444')

# print(r)

img = Image.open('London.png')
buffer = BytesIO()
img.save(buffer, format='PNG')
img_bytes = base64.b64encode(buffer.getvalue())
img_str = img_bytes.decode('utf-8')

s = {
    'input': img_str
}

r = requests.post('http://localhost:4444/test', data=json.dumps(s), headers={'content-type': 'application/json'})

print(r)