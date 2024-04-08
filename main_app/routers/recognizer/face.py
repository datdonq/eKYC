import os
import logging
from asyncio import Semaphore

import aiohttp
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile


error_logger = logging.getLogger("error_logger")
ai_logger = logging.getLogger("ai_logger")
io_logger = logging.getLogger("io_logger")

load_dotenv()

class FaceRecognizer:
    def __init__(self) -> None:
        self.router = APIRouter(tags=["Face Recognizer"])
        self.semaphore = Semaphore(10)

        async def limit_concurrency():
            async with self.semaphore:
                yield

        @self.router.post("/api/v1/recognizer/face")
        async def idcardreader(
            image: list[UploadFile],
            user_id: str = Form(...),
            _lock=Depends(limit_concurrency),
        ):
            try:
                image_contents = await image[0].read()
                face_contents = open(f"data/{user_id}/face_drop.jpg", "rb").read()
                data = {'file_face': image_contents, 'file_cmnd': face_contents}

                url = os.getenv("FACE_SIMILAR_API")
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=data) as response:
                        status = response.status
                        if status == 200:
                            data = await response.json()
                            if "confidence_score" in data:
                                return dict(
                                    status=data['status'],
                                    prediction=data['result_predict'],
                                    score=data['confidence_score']
                                )
                            return data
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
