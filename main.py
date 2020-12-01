import numpy as np
import pandas as pd
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import dataHandler
import dataLoader
from dataLoader import DataLoader
from sklearn.model_selection import train_test_split

# Global Variables
modelsDir = 'models'
modelFormats = ['h5']
inputImgsDir = 'data/final/imgs'
inputLabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpg']
labelFormats = ['npy']

K.set_image_data_format('channels_last')
smooth = 1.
RANDOM_STATE = 42

trainConfig = {
    'epochs' : 10,
    'batchSize' : 32,
    'modelStartNeurons' : 8,
    'dataLen': None, # Set during runtime
}

def genModelName(config):
    return('m%s-%s-%s-%s.h5'%(
        trainConfig['dataLen'],
        trainConfig['modelStartNeurons'],
        trainConfig['epochs'],
        trainConfig['batchSize']
    ))

def genModelConfig(modelName, verbose=True):
    x = [int(x) for x in modelName[1:-3].split('-')]
    if(len(x) != 4):
        print('Error: Incorrect model configuration format')
        exit()
    if(verbose):
        print('---------------------------------')
        print('Model Configuration: ')
        print('Length of dataset :  ',x[0])
        print('Starting neurons :   ',x[1])
        print('Total epochs:        ',x[2])
        print('Batch size:          ',x[3])
        print('---------------------------------')
    return({
        'dataLen': x[0],
        'modelStartNeurons': x[1],
        'epochs': x[2],
        'batchSize': x[3]
    })

# Calculate Dice coeff and loss to measure overlap between 2 samples
# Experimental
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == "__main__":
    imgNames = dataHandler.getImgNames(inputImgsDir, imgFormats)
    labelNames = dataHandler.getLabelNames(inputLabelsDir, labelFormats)
    
    if(len(imgNames) != len(labelNames)):
        print('Error: Image and Label lengths do not match')
        exit()
    
    trainConfig['dataLen'] = len(imgNames)
    print('Total dataset size:', trainConfig['dataLen'])

    # Train-validation split
    imgNamesTrain, imgNamesValidate, labelNamesTrain, labelNamesValidate = train_test_split(imgNames, labelNames, test_size=0.1, random_state=RANDOM_STATE)

    print('Train data size:   ',len(imgNamesTrain))
    print('Train label size:  ',len(labelNamesTrain))
    print('Test data size:    ',len(imgNamesValidate))
    print('Test label size:   ',len(labelNamesValidate))

    sampleImg = dataHandler.loadImg(inputImgsDir+'/'+imgNamesTrain[0])
    sampleLabel = dataHandler.loadNPArr(inputLabelsDir+'/'+labelNamesTrain[0])
    
    shImg = sampleImg.shape
    shLabel = sampleLabel.shape

    print('Image Dimensions:  ',shImg)
    print('Label Dimensions:  ',shLabel)

    # Initializing the data generators
    training_generator = DataLoader(
        imgNamesTrain, 
        labelNamesTrain, 
        inputImgsDir, 
        inputLabelsDir,
        trainConfig['batchSize'],
        shLabel[-1],
        (shImg[0], shImg[1])
    )
    validation_generator = DataLoader(
        imgNamesValidate, 
        labelNamesValidate, 
        inputImgsDir,
        inputLabelsDir,
        trainConfig['batchSize'],
        shLabel[-1],
        (shImg[0], shImg[1])
    )

    # Build the U-net model
    inputs = Input(shImg)

    # Starting neurons
    n = trainConfig['modelStartNeurons']

    # IMAGE SIZE 444 x 444
    conv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #pool4 = Dropout(0.20)(pool4)

    convMid = Conv2D(n*16, (3, 3), activation='relu', padding='same')(pool4)
    convMid = Conv2D(n*16, (3, 3), activation='relu', padding='same')(convMid)

    deConv4 = Conv2DTranspose(n*8, (3, 3), strides=(2, 2), padding='same')(convMid)
    upConv4 = concatenate([deConv4, conv4]) #, axis=3
    #upConv4 = Dropout(0.2)(upConv4)
    upConv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(upConv4)
    upConv4 = Conv2D(n*8, (3, 3), activation='relu', padding='same')(upConv4)

    deConv3 = Conv2DTranspose(n*4, (3, 3), strides=(2, 2), padding='same')(upConv4)
    upConv3 = concatenate([deConv3, conv3]) #, axis=3
    #upConv3 = Dropout(0.2)(upConv3)
    upConv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(upConv3)
    upConv3 = Conv2D(n*4, (3, 3), activation='relu', padding='same')(upConv3)

    deConv2 = Conv2DTranspose(n*2, (3, 3), strides=(2, 2), padding='same')(upConv3)
    upConv2 = concatenate([deConv2, conv2]) #, axis=3
    #upConv2 = Dropout(0.2)(upConv2)
    upConv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(upConv2)
    upConv2 = Conv2D(n*2, (3, 3), activation='relu', padding='same')(upConv2)

    deConv1 = Conv2DTranspose(n*1, (3, 3), strides=(2, 2), padding='same')(upConv2)
    upConv1 = concatenate([deConv1, conv1]) #, axis=3
    #upConv1 = Dropout(0.2)(upConv1)
    upConv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(upConv1)
    upConv1 = Conv2D(n*1, (3, 3), activation='relu', padding='same')(upConv1)

    outputLayer = Conv2D(shLabel[-1], (1), padding='same', activation='softmax')(upConv1)

    model = Model(inputs=[inputs], outputs=[outputLayer])
    model.compile(optimizer=Adam(lr = 5*(1e-4)), loss='categorical_crossentropy',metrics=['accuracy'])

    model.summary()

    callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    ]
    
    modelName = genModelName(trainConfig)
    print('Training model',modelName)
    genModelConfig(modelName, verbose=True)

    results = model.fit(
        training_generator, 
        validation_data=validation_generator,
        workers = 6,
        callbacks=callbacks,
        epochs=trainConfig['epochs']
    )
    
    keras.models.save_model(
        model=model,
        filepath= modelsDir+'/'+modelName,
    )

    print('Model',modelName,'saved to disk')

    '''
    Try:
        Batch Normalization when building model
    '''
