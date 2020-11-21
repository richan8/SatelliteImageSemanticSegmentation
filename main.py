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
import dataHandler
import dataLoader
from dataLoader import DataLoader

# Global Variables
inputImgsDir = 'data/final/imgs'
inputLabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
labelFormats = ['npy']
imgNames = [x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in labelFormats]
K.set_image_data_format('channels_last')
IMG_HEIGHT = None
IMG_WIDTH = None
smooth = 1.

# TRUNCATING THE DATASET FOR TESTING
imgNames = imgNames[:300]
labelNames = labelNames[:300]

# Calculate Dice coeff and loss to measure overlap between 2 samples
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

'''
### Testing the label image to vector conversion
label = cv2.imread('data/final/labels/label-0-0.jpeg', cv2.IMREAD_GRAYSCALE)
labelVec = vectorizeLabelImg(label)
regenLabel = labelVecToImg(labelVec)
print(compareImgs(label,regenLabel))
'''

if __name__ == "__main__":
    ## ------------ tensorflow u-net image segmentation ------------------ ##

    sampleImg = cv2.imread(inputImgsDir+'/'+imgNames[0])
    sh = sampleImg.shape
    imgs = np.zeros((len(imgNames),sh[0],sh[1],sh[2]))
    labels = np.zeros((len(labelNames),sh[0],sh[1],3))

    # load the entire dataset
    '''
    i = 0

    for imgName,labelName in zip(imgNames,labelNames):
        imgs[i] = cv2.imread(inputImgsDir+'/'+imgName)
        labels[i] = dataHandler.loadNPArr(inputLabelsDir+'/'+labelName)
        i += 1
    '''
    training_generator = DataLoader(imgNames, labelNames, inputImgsDir, inputLabelsDir)
    validation_generator = DataLoader(imgNames, labelNames, inputImgsDir, inputLabelsDir)
    print('Data Loaded')

    #IMG_HEIGHT = imgs.shape[1]
    #IMG_WIDTH = imgs.shape[2]
    IMG_HEIGHT = 464
    IMG_WIDTH = 464

    # Build the U-net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    # Starting neurons
    n = 3

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

    outputLayer = Conv2D(3, (1, 1), padding='same', activation='sigmoid')(upConv1)

    model = Model(inputs=[inputs], outputs=[outputLayer])
    model.compile(optimizer=Adam(lr = 1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()

    #results = model.fit(imgs,labels, validation_split=0.1, batch_size=32, epochs=3)
    results = model.fit(training_generator, validation_data=validation_generator)
    keras.models.save_model(
        model=model,
        filepath='models/m2.h5',
    )