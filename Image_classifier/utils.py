import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import argparse
import logging
from PIL import Image
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# TODO: Create the process_image function
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def preprocess(image_path):


    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224,224))
    image = normalize(image)
    image = image.numpy()
    return image


def prediction(ps, class_names, top_k):
    
    
    idx = np.argpartition(ps[0], -top_k)
    
    probs = ps[0][idx[-top_k:]]
    classes = idx[-top_k:]

    class_name = list()

    for i in range (0,top_k):

        class_name.append(class_names[str(classes[i] +1)])

    
    
    
    return probs, class_name