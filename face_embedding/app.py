####default lib
import os
import logging
import datetime
####need install lib
import uvicorn
####custom modules
from routers.face_detect_embedding import FaceDetectEmbedding
####default lib
from fastapi import FastAPI
from configparser import ConfigParser
import logging.config
import json

####
now = datetime.datetime.now()
#######################################
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
WORKER_NUM = str(config.get('main', 'WORKER_NUM'))
#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging_config_path = "config/logging.json"
if os.path.exists(logging_config_path):
    with open(logging_config_path, 'r') as f:
        config = json.load(f)
    logging.config.dictConfig(config)

io_logger = logging.getLogger("io_logger")
error_logger = logging.getLogger("error_logger")

#######################################
app= FastAPI(
            title="Face Detect-Embedding",
            docs_url="/"
        )
app.include_router(FaceDetectEmbedding().router)
print("API READY")
print("PORT",SERVICE_PORT)
#######################################
if __name__ == '__main__':
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())
    io_logger.debug("Start application.")
    uvicorn.run("app:app", port=SERVICE_PORT, host=SERVICE_IP,reload=True, log_level="debug")