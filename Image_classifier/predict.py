import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import argparse
from utils import preprocess
from utils import prediction
from PIL import Image
import tensorflow_hub as hub
from tensorflow.keras import regularizers
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path",default="./test_images/hard-leaved_pocket_orchid.jpg",
    help="path to image")
ap.add_argument("-m", "--model", type=str, default="best_model.h5",
    help="path to image classifier")
ap.add_argument("-t", "--top_k", type = int, default=1,
    help="threshold to top classes")
ap.add_argument("-c", "--category_names", type = str, default="label_map.json",
    help="map prediction to class names")


def main() :

    args = vars(ap.parse_args())

    image_path = args['image_path']
    model = args['model']
    top_k = args['top_k']
    category_names = args['category_names']


    with open(category_names, 'r') as f:
        class_names = json.load(f)




    reloaded_SavedModel = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})

    image = preprocess(image_path)

    image = np.expand_dims(image, axis = 0)

    ps = reloaded_SavedModel(image, training=False).numpy()

    if top_k > 1:

        preds, classes = prediction(ps,class_names,top_k)

        print("Top ", top_k, " Classes are:- ")

        for i in range (0,top_k):

            print(classes[i], " with probability of ",preds[i])

    else:

        class_idx = np.argmax(ps[0])

        Class_name = class_names[str(class_idx+1)] 

        print(" The given image is ", Class_name, " with probability of ", max(ps[0]))



if __name__ == '__main__' :
    main()
