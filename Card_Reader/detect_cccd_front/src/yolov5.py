import argparse
import sys
import time
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from utils.torch_utils import select_device


class LoadImages1:  # for inference
    def __init__(self, img, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.img = img
        self.nf = 1
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        self.count += 1
        img0 = self.img

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return "", img, img0, self.cap

    def __len__(self):
        return self.nf  # number of files

def load_model(model_path, imgsz):
    device=''
    device = select_device(device)
    half=False
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    return model, names, stride, device, half

def predict(model, names, imgsz, stride, device, half, process_img):
    augment=False
    visualize=False
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    agnostic_nms=False
    classes=None

    dataset = LoadImages1(process_img, img_size=imgsz, stride=stride)

    return_data = []
    for path, img, im0s, vid_cap in dataset:
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = model(img, augment=augment, visualize=visualize)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det_cpu = det.cpu().tolist()
#                print(det_cpu)
                # Write results
                for *xyxy, conf, cls in reversed(det_cpu):
                    bbox = xyxy
                    conf = float(conf)
                    cls = names[int(cls)]
                    return_data.append({'bbox': bbox, 'score': conf, 'cls': cls})

    return return_data


if __name__=="__main__":
    imgsz=640
    model_path = "../../room-3/idcard/yolov5_fcmtnd_050821/best.pt"
    fpath = "2eb9d47e763c4e52bf28fc515050f3b8_cmtnd_front.jpg"
    model, names, stride, device, half = load_model(model_path, imgsz)
    process_img = cv2.imread(fpath)
    start_time = time.time()
    return_data = predict(model, names, imgsz, stride, device, half, process_img)
    print("time", time.time()-start_time)
    
    for obj in return_data[:]:
        bbox = [int(i) for i in obj['bbox']]
        score = obj['score']
        cls = obj['cls']
        cv2.rectangle(process_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 3)
        
        process_img = Image.fromarray(process_img)                
            
        ttf=ImageFont.truetype('TimesNewRoman.ttf', 30)
        ImageDraw.Draw(process_img).text((bbox[0],bbox[1]-10), cls, fill=(0,0,255), font=ttf)
        
        process_img = np.asarray(process_img)
    
    cv2.imwrite("out.jpg", process_img)






