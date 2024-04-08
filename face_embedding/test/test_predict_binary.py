import base64
import urllib.parse
import requests
import json
import timeit
import sys
import io
import cv2
import numpy as np
import time
start_time = time.time()
#url = 'http://0.0.0.0:2336/predict_binary'
url = 'https://aiclub.uit.edu.vn/gpu/service/mtcnn_facenet/predict_binary'
####################################
file_path = "test.jpg"
####################################
f = {'binary_file': open(file_path, 'rb')}
####################################
response = requests.post(url, files = f)
response = response.json()
print(response)
print('time', time.time()-start_time)

img = cv2.imread(file_path)
for result in response['predicts'][0]:
    bbox = result['bbox']
    print(bbox)
    cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), (255,0,0), 2)
cv2.imwrite("out.jpg", img)
