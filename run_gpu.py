import os
import time
import requests
import openslide
from model import ProstateInferencer

URL = os.getenv("API_SERVER")
model = ProstateInferencer()

os.makedirs("data", exist_ok=True)

while True:
    time.sleep(1)
    r = requests.get("http://{}:8001/worker/get_task".format(URL))

    if r.headers['content-type'] == 'application/json':
        time.sleep(1)
        continue

    request_id = r.headers['content-disposition']
    path = "data/{}".format(request_id)
    with open(path, 'wb') as f:                                                 
        f.write(r.content)

    slide = openslide.OpenSlide(path)
    result = model.predict(slide)
    result['request_id'] = request_id
    requests.post("http://{}:8001/worker/post_result".format(URL), json=result)

