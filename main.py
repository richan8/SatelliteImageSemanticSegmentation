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
inputImgsDir = 'data/final/imgs'
inputLabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
imgNames = [x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in imgFormats]
K.set_image_data_format('channels_last')
IMG_HEIGHT = None
IMG_WIDTH = None
smooth = 1.

# Calculate Dice coeff and loss to measure overlap between 2 samples
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == "__main__":

    '''
    # sample img
    gray = []
    im = np.random.randint(256, size=(5,256,256,3))
    im_type = np.array(im, dtype=np.uint8)
    for i in range(im.shape[0]):
        gray.append(cv2.cvtColor(im_type[i], cv2.COLOR_RGB2GRAY))
    gray_img = np.array(gray, dtype=np.float32)
    #print(gray_img.shape)
    '''

    ## ------------ tensorflow u-net image segmentation ------------------ ##

    imgs = []
    labels = []

    # load the entire dataset
    for imgName,labelName in zip(imgNames,labelNames):
        imgDir = inputImgsDir+'/'+imgName
        labelDir = inputLabelsDir+'/'+labelName

        img = cv2.imread(imgDir)
        label = cv2.imread(labelDir, cv2.IMREAD_GRAYSCALE)

        '''
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.imshow('label',label)
        cv2.waitKey(0)
        '''

        imgs.append(img)
        labels.append(label)

    IMG_HEIGHT = imgs[0].shape[0]
    IMG_WIDTH = imgs[0].shape[1]

    # Random training on n samples
    N = 100
    indexes = np.random.choice(range(len(imgs)), replace=False, size=N)
    imgSamples = np.array(imgs)[indexes]
    labelSamples = np.array(labels)[indexes]

    # Build the U-net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    # Starting neurons
    n = 32

    # IMAGE SIZE 444 x 444
    conv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.20)(pool4)

    convMid = Conv2D(n*16, (3, 3), activation='relu', padding='same')(pool4)
    convMid = Conv2D(n*16, (3, 3), activation='relu', padding='same')(convMid)

    deConv4 = Conv2DTranspose(n*8, (3, 3), strides=(2, 2), padding='same')(convMid)
    upConv4 = concatenate([deConv4, conv4]) #, axis=3
    upConv4 = Dropout(0.2)(upConv4)
    upConv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(upConv4)
    upConv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(upConv4)

    deConv3 = Conv2DTranspose(n*4, (3, 3), strides=(2, 2), padding='same')(upConv4)
    upConv3 = concatenate([deConv3, conv3]) #, axis=3
    upConv3 = Dropout(0.2)(upConv3)
    upConv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(upConv3)
    upConv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(upConv3)

    deConv2 = Conv2DTranspose(n*2, (3, 3), strides=(2, 2), padding='same')(upConv3)
    upConv2 = concatenate([deConv2, conv2]) #, axis=3
    upConv2 = Dropout(0.2)(upConv2)
    upConv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(upConv2)
    upConv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(upConv2)

    deConv1 = Conv2DTranspose(n*1, (3, 3), strides=(2, 2), padding='same')(upConv2)
    upConv1 = concatenate([deConv1, conv1]) #, axis=3
    upConv1 = Dropout(0.2)(upConv1)
    upConv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(upConv1)
    upConv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(upConv1)

    outputLayer = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(upConv1)

    model = Model(inputs=[inputs], outputs=[outputLayer])
    #model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    #results = model.fit(im, gray_img, validation_split=0.1, batch_size=4, epochs=20)