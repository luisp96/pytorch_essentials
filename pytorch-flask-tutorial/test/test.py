import requests

# we send one POST request to our localhost flask app, with a file
resp = requests.post('http://localhost:5000/predict', files={'file': open('three.png', 'rb')}) #open the file seven.png in read binary mode

print(resp.text)