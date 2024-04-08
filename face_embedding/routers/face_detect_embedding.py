import logging
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from dotenv import load_dotenv
import numpy as np
import timeit
import rcode
import cv2
import os
import tensorflow as tf
import uuid
from configparser import ConfigParser
import src.detect_face as detect_face
import src.extracter as extracter

load_dotenv()

error_logger = logging.getLogger("error_logger")
ai_logger = logging.getLogger("ai_logger")
io_logger = logging.getLogger("io_logger")
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
PATH_EMBEDDING = str(config.get('main', 'PATH_EMBEDDING'))
WORKER_NUM = str(config.get('main', 'WORKER_NUM'))
PATH_TRUE = str(config.get('main','PATH_TRUE'))
PATH_FALSE = str(config.get('main','PATH_FALSE'))
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


class FaceDetectEmbedding:
    def __init__(self) -> None:
        self.router = APIRouter(tags=["Service_name"])

        # load model detect
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
                self.pnet = pnet
                self.rnet = rnet
                self.onet = onet
        ####LOAD MODEL EMBEDDINg
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 50
        config.inter_op_parallelism_threads = 5
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        extracter.restore_facenet_model(PATH_EMBEDDING)
    
        @self.router.post('/api/v1/face_detect_embedding')
        async def predict_face( file_cmnd: UploadFile = File(...), 
                            file_face: UploadFile = File(...)):
            ###################
            #####
            io_logger.debug("face detect embedding")
            return_result = {'code': '1001', 'status': rcode.code_1001}
            ###################
            try:
                start_time = timeit.default_timer()
                predicts = []
                try:
                    bytes_cmnd = await file_cmnd.read()
                    bytes_face = await file_face.read()
                except Exception as e:
                    error_logger.debug(e)
                    return_result = {'code': '609', 'status': rcode.code_609}
                    return; 
                ###########################
                # for file cmnd
                nparr_cmnd = np.fromstring(bytes_cmnd, np.uint8)
                process_image_cmnd = cv2.imdecode(nparr_cmnd, flags=1)
                # for face
                nparr_face = np.fromstring(bytes_face, np.uint8)
                process_image_face = cv2.imdecode(nparr_face, flags=1)
                ####opencv img to pillow
        #        process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        #        process_image = Image.fromarray(process_image)
                ####
                io_logger.debug("Step1 detect face")
                # predict for cmnd
                mtcnn_cmnd, _ = detect_face.detect_face(
                                process_image_cmnd, minsize,
                                self.pnet, self.rnet, self.onet,
                                threshold, factor)
                # predict for cmnd
                mtcnn_face, _ = detect_face.detect_face(
                                process_image_face, minsize,
                                self.pnet, self.rnet, self.onet,
                                threshold, factor)
                # if there are too many faces in one image then raise error
                if len(mtcnn_cmnd) != 1 or len(mtcnn_face) != 1:
                    io_logger.debug(rcode.code_703, exc_info=True)
                    return_result = {'code': '703', 'status': rcode.code_703}
                    return return_result
                # get bb of image cmnd
                bbox_cmnd = [int(i) for i in mtcnn_cmnd[0][:4]]
                crop_image_cmnd = process_image_cmnd[bbox_cmnd[1]: bbox_cmnd[3], bbox_cmnd[0]: bbox_cmnd[2]]
                # cv2.imwrite('./cmnd.jpg', crop_image_cmnd)
                # get bb of image face
                bbox_face = [int(i) for i in mtcnn_face[0][:4]]
                crop_image_face = process_image_face[bbox_face[1]: bbox_face[3], bbox_face[0]: bbox_face[2]]
                # cv2.imwrite('./face.jpg', crop_image_face)

                # embedding
                io_logger.debug("Step2 embedding face")
                ## cmnd
                h,w,_ = crop_image_cmnd.shape
                embedding_cmnd = extracter.extract_feature(self.sess, crop_image_cmnd, [(0,0,h,w)])
                ## face
                h,w,_ = crop_image_face.shape
                embedding_face = extracter.extract_feature(self.sess, crop_image_face, [(0,0,h,w)])

                cosine_similarity = tf.keras.losses.cosine_similarity(embedding_cmnd, embedding_face, axis=1)
                cosine_similarity_value = None
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                with tf.compat.v1.Session() as sess:
                    cosine_similarity_value = self.sess.run(cosine_similarity)

                if cosine_similarity_value >= 0.6:
                    path_foder = os.path.join(PATH_TRUE, uuid.uuid1().hex)

                    while not os.path.exists(path_foder):
                        path_foder = os.path.join(PATH_TRUE, uuid.uuid1().hex)
                        os.mkdir(path_foder)

                    path_image_cmnd = os.path.join(path_foder, f'{uuid.uuid1().hex}_cmnd.jpg')
                    path_image_face = os.path.join(path_foder, f"{uuid.uuid1().hex}_face.jpg")
                    return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': f"{cosine_similarity_value[0]}",
                                'process_time': timeit.default_timer()-start_time, 'save_at': path_foder,'result_preditc':True,
                                'WORKER_NUM': WORKER_NUM}
                    # save image
                    cv2.imwrite(path_image_cmnd, process_image_cmnd)
                    cv2.imwrite(path_image_face, process_image_face)

                else:
                    path_foder = os.path.join(PATH_FALSE, uuid.uuid1().hex)
                    while not os.path.exists(path_foder):
                        path_foder = os.path.join(PATH_FALSE, uuid.uuid1().hex)
                        os.mkdir(path_foder)

                    path_image_cmnd = os.path.join(path_foder, f'{uuid.uuid1().hex}_cmnd.jpg')
                    path_image_face = os.path.join(path_foder, f"{uuid.uuid1().hex}_face.jpg")

                    return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': f"{cosine_similarity_value[0]}",
                                'process_time': timeit.default_timer()-start_time, 'save_at': path_foder,'result_preditc':False,
                                'WORKER_NUM': WORKER_NUM}
                    # save image
                    cv2.imwrite(path_image_cmnd, process_image_cmnd)
                    cv2.imwrite(path_image_face, process_image_face)

            except Exception as e:
                    error_logger.debug(e)
                    return_result = {'code': '1001', 'status': rcode.code_1001}
            finally:
                return return_result