####default lib
import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
####need install lib
import torch
import uvicorn
import cv2
import traceback
import asyncio
import numpy as np
####custom modules
import rcode
####default lib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from configparser import ConfigParser
####
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
####
now = datetime.datetime.now()
#######################################
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))
WORKER_NUM = str(config.get('main', 'WORKER_NUM'))
MODEL_PATH = str(config.get('main', 'MODEL_PATH'))
#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w",
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
#######################################
####LOAD MODEL HERE
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = MODEL_PATH
config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['predictor']['beamsearch'] = False
config['seq_modeling'] = 'transformer'
predictor = Predictor(config)
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("WORKER_NUM", WORKER_NUM)
print("API READY")
#######################################
@app.post('/predict')
async def predict(data: PredictData):
    ###################
    #####
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            images = jsonable_encoder(data.images)
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        for image in images:
            image_decoded = base64.b64decode(image)
            jpg_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
            process_image = cv2.imdecode(jpg_as_np, flags=1)
            ####opencv img to pillow
            process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            process_image = Image.fromarray(process_image)
            ####
            predict = predictor.predict(process_image)
            predicts.append([str(predict), 1.0])
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_binary_batch')
async def predict_binary_batch(binary_files: List[UploadFile] = File(...)):
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}

    try:
        start_time = timeit.default_timer()
        predicts = []
        for binary_file in binary_files:
            try:
                bytes_file = await binary_file.read()
            except Exception as e:
                logger.error(e, exc_info=True)
                return_result = {'code': '609', 'status': rcode.code_609}
                return return_result

            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            ####opencv img to pillow
            process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            process_image = Image.fromarray(process_image)
            ####
            predict = predictor.predict(process_image)
            predicts.append([str(predict), 1.0])

        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
    except Exception as e:
        logger.error(e, exc_info=True)
        return_result = {'code': '1001', 'status': rcode.code_1001}

    finally:
        return return_result

@app.post('/predict_binary')
async def predict_binary(binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        ####opencv img to pillow
        process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        process_image = Image.fromarray(process_image)
        ####
        predict = predictor.predict(process_image)
        predicts.append([str(predict), 1.0])
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result
        
@app.post('/predict_multi_binary')
async def predict_binary(binary_files: Optional[List[UploadFile]] = File(...)):
    ###################
    #####
    logger.info("predict_multi_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file_list = []
            for binary_file in binary_files:
                bytes_file_list.append(await binary_file.read())
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        process_image_list = []
        for bytes_file in bytes_file_list:
            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            ####opencv img to pillow
            process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            process_image = Image.fromarray(process_image)
            ####
            process_image_list.append(process_image)  
        predict_strs, predict_confs = predictor.predict_batch(process_image_list, return_prob=True)
        for predict_str, predict_conf in zip(predict_strs, predict_confs):
            predicts.append({"str": str(predict_str), "conf": predict_conf})
            
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
        logger.info(return_result)
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_multipart')
async def predict_multipart(argument: Optional[float] = Form(...),
                binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_multipart")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        ####opencv img to pillow
#        process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
#        process_image = Image.fromarray(process_image)
        ####
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result
        
@app.get('/info')
async def info():
    ###################
    #####
    logger.info("info")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        ####opencv img to pillow
#        process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
#        process_image = Image.fromarray(process_image)
        ####
        info = "vietocr service; model: pretrain-transformerocr.pth; install by pip3."
        return_result = {'code': '1000', 'status': rcode.code_1000, 'info': info,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'dict'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP, reload=True)
    # uvicorn.run("main:APIer", host="0.0.0.0", port=8002, reload=True, log_level="debug")

