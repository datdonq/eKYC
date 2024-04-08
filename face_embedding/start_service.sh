export PYTHONIOENCODING=UTF-8
export LANG=C.UTF-8
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/mtcnn_facenet/mtcnn_facenet
export PYTHONPATH=${PYTHONPATH}:/mtcnn_facenet/mtcnn_facenet
export PYTHONPATH=${PYTHONPATH}:/mtcnn_facenet/mtcnn_facenet/src
uvicorn service:app --port 80 --host 0.0.0.0 --workers 1
