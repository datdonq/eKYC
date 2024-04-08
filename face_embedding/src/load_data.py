import faiss
from tqdm import tqdm, tqdm_notebook
import os
import numpy as np

def wfile(root):
    img_fps = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name == 'face.npy':
                img_fps.append(os.path.join(path, name))

    return sorted(img_fps)
def load_data(path: str = "./data"):
    image_fps = wfile('../data')
    
    id2img_fps = dict(enumerate(image_fps))

if __name__=='__main__':
    load_data()
    