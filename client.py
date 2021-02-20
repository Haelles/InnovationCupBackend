import time
import requests

root = "./api/"
url = "http://127.0.0.1:5000/generate"
files = {'original': open(root + 'original.jpg', 'rb'),
         'sketch': open(root + 'sketch.jpg', 'rb'),
         'mask': open(root + 'mask.jpg', 'rb'),
         'stroke': open(root + 'stroke.jpg', 'rb')}
start_time = time.time()
resp = requests.post(url, files=files)
print(resp)
print(resp.json())
end_time = time.time()

print("cost time {}s".format(end_time-start_time))
