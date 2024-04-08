import logging
from asyncio import Semaphore
import numpy as np
import cv2
import os

import aiohttp
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from io import BytesIO

from src.processing.post_processing import (
    get_text_from_class,
    get_text_from_classes,
    get_base64_face_image,
    convert_date_format,
)

from routers.ocrcard.utils import sub_function

error_logger = logging.getLogger("error_logger")
ai_logger = logging.getLogger("ai_logger")
io_logger = logging.getLogger("io_logger")

load_dotenv()


class OCRPassportReader:
    def __init__(self) -> None:
        self.router = APIRouter(tags=["OCR Reader"])
        self.semaphore = Semaphore(10)

        async def limit_concurrency():
            async with self.semaphore:
                yield


        @self.router.post("/api/v1/reader/passport")
        async def passportcardreader(
            image: UploadFile = File(...),
            user_id: str = Form(...),
            _lock=Depends(limit_concurrency),
        ):
            try:
                image_contents = await image.read()

                data = {'binary_file': image_contents}
                io_logger.debug("Received input data successfully.")

                url = os.getenv('PASSPORT_DETECT_FIELD_API')

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=data) as response:

                        status = response.status

                        if status == 200:
                            data = await response.json()
                            ai_logger.debug(f"Call api at {url} successfully")
                        else:
                            data = {}
                            ai_logger.debug(f"Call api at {url} failed. Status code {status}")
                            raise HTTPException(
                                status_code=202,
                                detail={
                                    "msg": "Detection API failed.",
                                    "content": "Pleae try again.",
                                }
                            )

                predicts = data["predicts"][0]
                # await sub_function.visualize_bboxes(image_contents, predicts, output_path="output_after_detect.png")
                print(predicts)

                if await sub_function.check_is_enough_2_conner(predicts=predicts):

                    ai_logger.debug("Image is enough quality.")

                    # NOTE: Dummy 
                    pil_image = sub_function.convert_byte_to_pil(image_contents)
                    face_box = [item for item in predicts if item['cls'] in ['face']][0]['bbox']
                    face_img = pil_image.crop(face_box)
                    os.makedirs(f"data/{user_id}", exist_ok=True)
                    face_img.save(f"data/{user_id}/face_drop.jpg")


                    # xóa thông tin face, conner and emblem
                    predicts = [item for item in predicts if item['cls'] not in ["midright", "midleft", 'face', 'emblem']]

                    url_vietocr, data = await sub_function.prepare_data_for_api_vietocr(sub_function.convert_byte_to_pil(image_contents), predicts)
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url_vietocr, data=data) as response_vietocr:
                            status = response_vietocr.status
                            if status == 200:
                                data_vietocr = await response_vietocr.json()  # get JSON data from response
                                for i in range(len(predicts)):
                                    predicts[i]["text"] = data_vietocr["predicts"][i]["str"]  # access data_vietocr here
                                ai_logger.debug(f"Call api at {url_vietocr} successfully")
                            else:
                                data_vietocr = {}
                                ai_logger.debug(f"Call api at {url_vietocr} failed. Status code {status}")

                    # await sub_function.visualize_bboxes(image_contents, predicts, vis_with="text", output_path="output_after_vietocr.png")
                    

                    
                    return {
                        "status": True,
                        "message": "Success",
                        "annotations": {
                            "nationality": get_text_from_class(class_name="cnationality", predict=predicts),
                            "date_of_expiry": get_text_from_class(class_name="cdoe", predict=predicts),
                            "data_of_birth": get_text_from_class(class_name="cdob", predict=predicts),
                            "id_passport": get_text_from_class(class_name="cppnum", predict=predicts),
                            "issued_date": get_text_from_class(class_name="cissue_date", predict=predicts),
                            "idcard_num": get_text_from_class(class_name="cidnum", predict=predicts),
                            "type": get_text_from_class(class_name="ctype", predict=predicts),
                            "sex": get_text_from_class(class_name="csex", predict=predicts),
                            "issue_place": get_text_from_class(class_name="cissue_place", predict=predicts),
                            "name": get_text_from_classes("cname1", "cname2", predicts),
                            "code": get_text_from_class(class_name="ccode", predict=predicts),
                            "place_of_birth": get_text_from_classes("chome1", "chome2", predicts),
                            "info": get_text_from_classes("hc_info1", "hc_info2", predict=predicts),
                        }
                    }
                else: 
                    ai_logger.debug("Image is not enough quality.")
                    return {
                        "status": False,
                        "message": "Image is not enough quality.",
                        "annotations": {}
                    }

            
            except Exception as err:
                error_logger.error(err)
                _default_content_error = "Sorry! Please try again!"
                raise HTTPException(
                    status_code=202,
                    detail={
                        "msg": "Failed",
                        "content": _default_content_error,
                    }
                )
