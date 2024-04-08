import requests
import cv2
import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw

start_time = time.time()

url = 'http://0.0.0.0:2820/predict_binary'

####################################
file_path = "test_2.jpg"
####################################
f = {'binary_file': open(file_path, 'rb')}
####################################

response = requests.post(url, files = f)
response = response.json()
print(response)
print('time', time.time()-start_time)

process_img = cv2.imread(file_path)
return_data = response['predicts']
for obj in return_data[:]:
    bbox = [int(i) for i in obj['bbox']]
    score = obj['score']
    cls = obj['cls']
    if score > 0.3:
        cv2.rectangle(process_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
        
        process_img = Image.fromarray(process_img)                
            
        ttf=ImageFont.truetype('TimesNewRoman.ttf', 30)
        ImageDraw.Draw(process_img).text((bbox[0],bbox[1]-10), cls, fill=(0,0,255), font=ttf)
        
        process_img = np.asarray(process_img)

cv2.imwrite(f"out_{file_path}.jpg", process_img)

