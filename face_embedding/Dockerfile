FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

WORKDIR /base

RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install numpy==1.16.1
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install configparser==5.0.2
RUN pip3 install six==1.16.0
RUN pip3 install future==0.18.2
RUN pip3 install python-multipart
RUN pip3 install uvicorn==0.14.0 fastapi==0.65.2
RUN pip3 install requests
RUN pip3 install protobuf==3.19.6
RUN pip3 install tqdm==4.64.1
RUN pip3 install faiss==1.5.3
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install python-dotenv==0.20.0

RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install numpy==1.16.1
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install configparser==5.0.2
RUN pip3 install six==1.16.0
RUN pip3 install future==0.18.2
RUN pip3 install python-multipart
RUN pip3 install uvicorn==0.14.0 fastapi==0.65.2
RUN pip3 install requests
RUN pip3 install protobuf==3.19.6
RUN pip3 install tqdm==4.64.1
RUN pip3 install faiss==1.5.3
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install python-dotenv==0.20.0
RUN pip3 install scipy==1.5.4
RUN pip3 install scikit-learn==0.24.2

CMD ["python3", "app.py"]

