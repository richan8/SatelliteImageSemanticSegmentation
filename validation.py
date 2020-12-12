import numpy as np
import cv2
import os
import tensorflow as tf
import keras
from main import dice_coef, dice_coef_loss
import dataHandler
from main import genModelConfig

# Global Variables
# Model 22e: m1260-8-22-32.h5
# Model 10e: m1260-8-10-32.h5
modelName = 'm16000-32-4-14.h5' # Edit this to select model
modelsDir = 'models'
imgsDir = 'data/val/imgs'
labelsDir = 'data/val/labels'
imgFormats = ['jpeg','png','jpeg']
labelFormats = ['npy']
imgNames = [x for x in sorted(os.listdir(imgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(labelsDir)) if x.split('.')[-1] in labelFormats]

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
        predVec = dataHandler.tensorToPrediction(pred, img[0], thresold=0.4)
        acc = dataHandler.compareImgs(labelVec,predVec)
        accuracies.append(acc)

        if(showImgs):
            predImg = dataHandler.labelVecToImg(predVec, 'RGB')
            cv2.imshow('Image: ',img[0])
            cv2.waitKey(0)
            print('Prediction Accuracy: %0.3f'%(acc))
            cv2.imshow('Prediction: ',predImg)
            cv2.waitKey(0)
            
        i += 1
        print('Validation in Progress:\t%0.2f%%'%(100*(i)/len(imgNames)))
    return(sum(accuracies)/len(accuracies))

if __name__ == "__main__":
    #samples = 10
    #sampleIndexes = np.random.randint(0,len(imgNames),size=(samples))
    #sampleImgNames = [imgNames[i] for i in sampleIndexes]
    #sampleLabelNames = [labelNames[i] for i in sampleIndexes]
    #valDir = 'data/val'
    #valImgNames = ['#3.png', '#1.png', '#2.png', '#4.png', '#5.png']
    #Shuffle the array
    shuffler = np.random.permutation(len(imgNames))
    imgNames = list(np.array(imgNames)[shuffler])[:100]
    labelNames = list(np.array(labelNames)[shuffler])[:100]

    acc = validate(imgsDir, labelsDir, imgNames, labelNames, modelName, showImgs=True)
    print('Model Accuracy: %0.3f'%(acc))