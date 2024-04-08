import os
import logging
from fastapi import HTTPException
import aiohttp
from io import BytesIO
from src.processing.pre_processing import preprocessing
from PIL import Image

from dotenv import load_dotenv
load_dotenv()
import numpy as np
import cv2


error_logger = logging.getLogger("error_logger")
ai_logger = logging.getLogger("ai_logger")
io_logger = logging.getLogger("io_logger")


class sub_function: 
    def __init__(self) -> None:
        pass

    @staticmethod
    async def get_url_api(type_card: int, is_front: int) -> str: 
        """
            Get the URL API based on the type of card and the side of the card.
            
            Args:
                type_card (int): The type of card. 0 for CCCD and 1 for CMTND.
                is_front (int): The side of the card. 1 for front and 0 for back.
            
            Returns:
                str: The URL API determined based on the type of card and the side of the card.
            
            Raises:
                HTTPException: If the URL API cannot be determined, an HTTPException is raised with the status code 202 and a detail message.
        """
        url = ""
        if type_card == 0: 
            url = os.getenv('FRONT_CCCD_DETECT_FIELD_API' if is_front else 'BACK_CCCD_DETECT_FIELD_API')
        elif type_card == 1: 
            url = os.getenv('FRONT_CMTND_DETECT_FIELD_API' if is_front else 'BACK_CMTND_DETECT_FIELD_API')
        
        if url: 
            ai_logger.debug(f"Already determined API url")
            return url 
        else: 
            ai_logger.debug(f"Can not determine API url")
            raise HTTPException(
                status_code=202,
                detail={
                    "msg": "Can not determine type card.",
                    "content": "Please try again.",
                }
            )
        
    @staticmethod
    async def check_is_enough_4_conner(predicts: list) -> bool:
        """
        Check if there are enough "topleft", "topright", "botright", and "botleft" classes in the list of predicts.

        Parameters:
            predicts (list): A list of dictionaries representing predictions.

        Returns:
            bool: True if there are enough "topleft", "topright", "botright", and "botleft" classes, False otherwise.
        """
        status = all(cls in [predict["cls"] for predict in predicts] for cls in ["topleft", "topright", "botright", "botleft"])
        return True if status else False
    
    @staticmethod
    async def check_is_enough_2_conner(predicts: list) -> bool:
        """
        Check if there are enough "midright" "midleft" classes in the list of predicts. (only for passport)

        Parameters:
            predicts (list): A list of dictionaries representing predictions.

        Returns:
            bool: True if there are enough "midright" and "midleft" classes, False otherwise.
        """
        # status = all(cls in [predict["cls"] for predict in predicts] for cls in ["midright", "midleft"])
        # return True if status else False
        
        # NOTE: Update dummy
        return True

    
    @staticmethod
    async def prepare_data_for_api_vietocr(pil_image: bytes, predicts: list) -> tuple:
        """
        Prepare data for the VietOCR API.

        Args:
            pil_image (bytes): image in PIL format
            predicts (list): The list of predicted bounding boxes.

        Returns:
            tuple: A tuple containing the URL of the VietOCR API and the prepared data.

        Raises:
            None
        """
        # list_image_bytes = [preprocessing.convert_PIL_to_bytes(Image.open(BytesIO(pil_image)).convert("RGB").crop(i["bbox"])) for i in predicts]
        list_image_bytes = []
        # pil_image = Image.open(BytesIO(pil_image)).convert("RGB")
        for idx, i in enumerate(predicts):
            crop_image = pil_image.crop(i["bbox"])
            # crop_image.save(f"image_crop_{idx}.png")
            list_image_bytes.append(preprocessing.convert_PIL_to_bytes(crop_image))
        url_vietocr = os.getenv('VIETOCR_RECOGNIZE_API')

        # Prepare aiohttp form data
        data = aiohttp.FormData()
        for i, img in enumerate(list_image_bytes):
            data.add_field('binary_files', img, filename='file{}.jpg'.format(i), content_type='application/octet-stream')

        return url_vietocr, data
    
    @staticmethod
    async def visualize_bboxes(image_contents, annotations, vis_with="cls",  output_path="output_image.jpg"):
        # Chuyển image_contents từ dạng byte sang mảng numpy
        image_nparr = np.frombuffer(image_contents, np.uint8)
        image = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)
        
        # Duyệt qua mỗi annotation và vẽ bbox
        for annotation in annotations:
            bbox = annotation['bbox']
            score = annotation['score']
            cls = annotation['cls'] 

            # Chuyển bbox sang dạng (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Vẽ bbox lên ảnh
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Thêm text vào ảnh
            if vis_with == "cls":
                label = f"{cls} {score:.2f}"
            else: 
                label = annotation['text']
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Lưu ảnh
        cv2.imwrite(output_path, image)

    @staticmethod
    def convert_byte_to_pil(byte_image) -> Image: 
        """
        Convert a byte image to a PIL Image.

        Parameters:
            byte_image (bytes): The byte representation of the image.

        Returns:
            Image: The PIL Image object.
        """
        opencv_image = cv2.imdecode(np.frombuffer(byte_image, np.uint8), cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        return pil_image