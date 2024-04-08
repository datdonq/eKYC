import json
import logging
import os

import uvicorn
from fastapi import FastAPI
import logging.config
from routers.ocrcard.ocr_id_card import ocr_id_card
from routers.ocrcard.ocr_passport import ocr_passport
from routers.cardreader.ocr_id_card import OCRIDCardReader
from routers.cardreader.ocr_passport import OCRPassportReader
from routers.recognizer.face import FaceRecognizer


logging_config_path = "configs/logging.json"
if os.path.exists(logging_config_path):
    with open(logging_config_path, 'r') as f:
        config = json.load(f)
    logging.config.dictConfig(config)

io_logger = logging.getLogger("io_logger")
error_logger = logging.getLogger("error_logger")


class PredictorAPI:
    def __init__(self) -> None:
        self.app = FastAPI(
            title="My API",
            docs_url="/"
        )
        
        self.app.include_router(OCRIDCardReader().router)
        self.app.include_router(OCRPassportReader().router)

        self.app.include_router(FaceRecognizer().router)
        
        self.app.include_router(ocr_id_card().router)
        self.app.include_router(ocr_passport().router)
        

APIer = PredictorAPI().app

if __name__=="__main__":
    io_logger.debug("Start application.")
    uvicorn.run(
        "app:APIer", 
        host=os.environ["APP_HOST"], 
        port=int(os.environ["APP_PORT"]), 
        reload=True, 
        log_level="debug"
    )