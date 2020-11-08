import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Global Variables
K.set_image_data_format('channels_last')
inputImgsDir = 'data/final/imgs'
inputLabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
imgNames = [x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in imgFormats]
smooth = 1.
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Calculate Dice coeff and loss to measure overlap between 2 samples
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == "__main__":

    # sample img
    gray = []
    im = np.random.randint(256, size=(5,256,256,3))
    im_type = np.array(im, dtype=np.uint8)
    for i in range(im.shape[0]):
        gray.append(cv2.cvtColor(im_type[i], cv2.COLOR_RGB2GRAY))
    gray_img = np.array(gray, dtype=np.float32)
    #print(gray_img.shape)

    ## ------------ tensorflow u-net image segmentation ------------------ ##

    img = []
    label = []
    # load the entire data
    for imgName,labelName in zip(imgNames,labelNames):
        imgDir = inputImgsDir+'/'+imgName
        labelDir = inputLabelsDir+'/'+labelName

        img.append(cv2.imread(imgDir))
        label.append(cv2.imread(labelDir, cv2.IMREAD_GRAYSCALE))

    # Random training on n samples
    N = 100
    indexes = np.random.choice(range(len(img)), replace=False, size=N)
    img_sample = np.array(img)[indexes]
    label_sample = np.array(label)[indexes]

    # Build the U-net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    results = model.fit(im, gray_img, validation_split=0.1, batch_size=4, epochs=20)