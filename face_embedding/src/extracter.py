from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
from scipy import misc
from src import facenet

img_size = 160

def restore_facenet_model(model_path):
    # load the model
    print('Loading feature extraction model')
    facenet.load_model(model_path)


def restore_classifier(classifier_path):
    # load the classifier
    print('Loading face classifier')
    if os.path.isdir(classifier_path):
        classifier = []
        class_names = []
        for root, dirs, files in os.walk(classifier_path):
            for file in files:
                with open(os.path.join(root, file), 'rb') as infile:
                    tmp1, tmp2 = pickle.load(infile)
                    classifier.append(tmp1)
                    class_names.append(tmp2)

    if os.path.isfile(classifier_path):
        with open(classifier_path, 'rb') as infile:
            classifier, class_names = pickle.load(infile)
            print("point_2", type(classifier))

    return classifier, class_names

def get_face_in_frame(frame, aligned_list):
    images = np.zeros((len(aligned_list), img_size, img_size, 3))
    i = 0
    for face_pos in aligned_list:
        if face_pos[0] < 0 or face_pos[1] < 0:
            continue
        else:
            img = frame[face_pos[1]:face_pos[3], face_pos[0]:face_pos[2], ]
            if img.ndim == 2:
                img = facenet.to_rgb(img)
#            img = misc.imresize(img, (img_size, img_size), interp='bilinear')
            img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
            img = facenet.prewhiten(img)
            img = facenet.crop(img, False, img_size)
            images[i, :, :, :] = img
            i += 1
    return images

def extract_feature(sess, frame, aligned_list):
        # get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    # run forward pass to calculate embeddings
    emb_array = np.zeros((len(aligned_list), embedding_size))
    faces = get_face_in_frame(frame, aligned_list)
    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    return emb_array
