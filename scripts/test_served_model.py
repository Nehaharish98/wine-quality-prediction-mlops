import requests
import json

wine_features = [
    [7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]
]

data = {
    "inputs": wine_features
}
headers = {"Content-Type": "application/json"}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    data=json.dumps(data),
    headers=headers
)

print("Response:", response.json())