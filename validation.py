import numpy as np
import cv2
import os
import tensorflow as tf
import keras
from main import dice_coef, dice_coef_loss
import dataHandler
from main import genModelConfig

# Global Variables
modelName = 'm1680-8-20-6.h5' # Edit this to select model
modelsDir = 'models'
ImgsDir = 'data/final/imgs'
LabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
labelFormats = ['npy']
imgNames = [x for x in sorted(os.listdir(ImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(LabelsDir)) if x.split('.')[-1] in labelFormats]

def validate(imgsDir,labelsDir,imgNames,labelNames,modelName,showImgs=False, showModelSummary=False):
    model = tf.keras.models.load_model(modelsDir+'/'+modelName)
    print('Loaded model',modelName)
    if(showModelSummary):
        model.summary()
    genModelConfig(modelName)

    accuracies = []
    i = 0
    for imgName,labelName in zip(imgNames,labelNames):
        imgPath = imgsDir+'/'+imgName
        labelPath = labelsDir+'/'+labelName

        img = np.array([dataHandler.loadImg(imgPath)])
        labelVec = dataHandler.loadNPArr(labelPath)
        label = dataHandler.labelVecToImg(labelVec)
        pred = model.predict(img)[0]
        predVec = dataHandler.tensorToPrediction(pred,thresold=0.4)
        acc = dataHandler.compareImgs(labelVec,predVec)
        accuracies.append(acc)

        if(showImgs):
            predImg = dataHandler.labelVecToImg(predVec, 'RGB')
            cv2.imshow('Image: ',img[0])
            cv2.waitKey(0)
            cv2.imshow('Label: ',label)
            cv2.waitKey(0)
            cv2.imshow('Prediction: ',predImg)
            cv2.waitKey(0)
            
        i += 1
        print('Validation in Progress:\t%0.2f%%'%(100*(i)/len(imgNames)))
    return(sum(accuracies)/len(accuracies))

if __name__ == "__main__":
    samples = 10
    sampleIndexes = np.random.randint(0,len(imgNames),size=(samples))
    sampleImgNames = [imgNames[i] for i in sampleIndexes]
    sampleLabelNames = [labelNames[i] for i in sampleIndexes]

    acc = validate(ImgsDir, LabelsDir, sampleImgNames, sampleLabelNames, modelName, showImgs=False)
    print('Model Accuracy: %0.3f'%(acc))