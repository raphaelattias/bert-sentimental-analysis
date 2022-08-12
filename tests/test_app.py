import requests

resp = requests.post("http://127.0.0.1:5000/predict", json={"file": "this was terrible"})

print(resp.json())