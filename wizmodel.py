import requests
import json
import base64
import time

api_key = ""

url = "https://api.wizmodel.com/sdapi/v1/txt2img"
prompt = "A nerd cat wearing glasses and solve math problems on the blackboard."

payload = json.dumps({
  "prompt": prompt,
  "steps": 100
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer '+ api_key
}

# count time 
start = time.time()

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json())

# Example base64 string (replace with your own)
base64_string = response.json()['images'][0]
print(base64_string)

with open("imageToSave.png", "wb") as fh:
    fh.write(base64.decodebytes(base64_string))

stop = time.time()

print(f"Time: {stop - start}s")