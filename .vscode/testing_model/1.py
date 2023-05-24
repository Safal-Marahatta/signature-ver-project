
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from tensorflow.keras.preprocessing import image
import pandas as pd
import itertools
import sklearn
# this is for submitted image to array
folder = "C:/Users/safal/minor project/Signature-Verification--2/media"
labl=0


def load_images_from_folder(folder_path):
    for image in os.listdir(folder_path):
        img_arr = cv2.imread(os.path.join(folder, image), cv2.IMREAD_GRAYSCALE)
        img_arr = img_arr/255
        new_arr = cv2.resize(img_arr, (70, 70))
        return new_arr


img = load_images_from_folder(folder)
#plt.imshow(img,cmap='gray')

# this for customer id to image convert
DATADIR = "C:/Users/safal/Downloads/archive (3)/sign_data/sign_data/train"
CATEGORIES = ['001', '002', '003']
data = []  # this is a list
labl = []


def create_image_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        # path to 001 or 001_forg or 002 ----dir
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(
                path, img), cv2.IMREAD_GRAYSCALE)
            img_array = img_array/255
            new_array = cv2.resize(img_array, (70, 70))
            # print(new_array)
            data.append([new_array])
            labl.append(class_num)


create_image_data()
labl = np.array(labl)
images = np.array(data)
image=images[55]

siamese_model = load_model('3person.h5')
verification = siamese_model.predict(
    [image.reshape((1, 70, 70)), img.reshape((1, 70, 70))]).flatten()
verification = verification*100
print(verification[0])
