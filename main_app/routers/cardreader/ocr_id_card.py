import os
import logging
from asyncio import Semaphore

import aiohttp
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from routers.ocrcard.utils import sub_function
from src.processing.post_processing import (convert_date_format,
                                            get_text_from_class,
                                            get_text_from_classes)

error_logger = logging.getLogger("error_logger")
ai_logger = logging.getLogger("ai_logger")
io_logger = logging.getLogger("io_logger")

load_dotenv()

class OCRIDCardReader:
    def __init__(self) -> None:
        self.router = APIRouter(tags=["OCR Reader"])
        self.semaphore = Semaphore(10)

        async def limit_concurrency():
            async with self.semaphore:
                yield

        @self.router.post("/api/v1/reader/idcard")
        async def idcardreader(
            image: UploadFile = File(...),
            type_card: int = Form(...),
            is_front: int = Form(...),
            user_id: str = Form(...),
            _lock=Depends(limit_concurrency),
        ):
            io_logger.debug("Step 1: Received input data successfully.")
            try:
                image_contents = await image.read()
                data = {'binary_file': image_contents}
                io_logger.debug("Step 1: Received input data successfully.")
                url = await sub_function.get_url_api(type_card, is_front)
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
                predicts = data["predicts"]
                # =================================================================
                # Vẽ bbox ra coi đúng không?
                # await sub_function.visualize_bboxes(image_contents, predicts, output_path="output_after_detect.png")
                # =================================================================
                if await sub_function.check_is_enough_4_conner(predicts):
                    ai_logger.debug("Image is enough quality.")
                    # face_base64_string = await get_base64_face_image(image_contents, predicts)
                    pil_image = sub_function.convert_byte_to_pil(image_contents)
                    if is_front:
                    # NOTE: Dummy 
                        face_box = [item for item in predicts if item['cls'] in ['face']][0]['bbox']
                        face_img = pil_image.crop(face_box)
                        os.makedirs(f"data/{user_id}", exist_ok=True)
                        face_img.save(f"data/{user_id}/face_drop.jpg")

                    predicts = [item for item in predicts if item['cls'] not in ['topleft', 'botright', 'botleft', 'topright', 'face', 'emblem']]

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
                    
                    # print(predicts)
                    if is_front:
                        output = {
                            "status": True,
                            "message": "Success",
                            "annotations": {
                                "place_of_origin": get_text_from_classes("chome1", "chome2", predicts),
                                "place_of_residence": get_text_from_classes("caddress1", "caddress2", predicts),
                                "nationality": get_text_from_class(class_name="cnationality", predict=predicts),
                                "name": get_text_from_classes("cname1", "cname2", predicts),
                                "id": get_text_from_class(class_name="cidnum", predict=predicts),
                                "date_of_birth": get_text_from_class(class_name="cdob", predict=predicts),
                                "sex": get_text_from_class(class_name="csex", predict=predicts),
                                "data_of_expiry": get_text_from_class(class_name="cdoe", predict=predicts),
                                # "face": face_base64_string
                            }
                        }
                    else:
                        output = {
                            "status": True,
                            "message": "Success",
                            "annotations": {
                                "issued_date": get_text_from_class(class_name="cissue_date", predict=predicts) if type_card == 0 else convert_date_format(get_text_from_class(class_name="cissue_date", predict=predicts))
                            }
                        }

                    return output

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
