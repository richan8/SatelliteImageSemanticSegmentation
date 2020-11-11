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
from main import dice_coef, dice_coef_loss

# Global Variables
inputImgsDir = 'data/final/imgs'
inputLabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
imgNames = [x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in imgFormats]

# load model
loadedModel = tf.keras.models.load_model('models/m1.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
loadedModel.summary()

# validation(just 2 images for now)
imgTest = []
labelTest = []
imgDir = []
labelDir = []
for i in range(2):
    imgDir.append(inputImgsDir+'/'+imgNames[i])
    labelDir.append(inputLabelsDir+'/'+labelNames[i])
for i in range(2):
    imgTest.append(cv2.imread(imgDir[i]))
    labelTest.append(cv2.imread(labelDir[i], cv2.IMREAD_GRAYSCALE))

label_pred = loadedModel.predict(np.array(imgTest))
for l in label_pred:
    print("Label: ",label_pred[i])
    cv2.imshow("label", label_pred[i])
    cv2.waitKey(0)