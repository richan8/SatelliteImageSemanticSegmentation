import numpy as np
import cv2
import os
import tensorflow as tf
import keras
from main import dice_coef, dice_coef_loss
import dataHandler

# Global Variables
ImgsDir = 'data/final/imgs'
LabelsDir = 'data/final/labels'
imgFormats = ['jpeg','png','jpeg']
labelFormats = ['npy']
imgNames = [x for x in sorted(os.listdir(ImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(LabelsDir)) if x.split('.')[-1] in labelFormats]

# load model
#loadedModel = tf.keras.models.load_model('models/m9.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
loadedModel = tf.keras.models.load_model('models/m9.h5')
loadedModel.summary()

def validate(imgsDir,labelsDir,imgNames,labelNames,model, showImgs=False):
    accuracies = []
    i = 0
    for imgName,labelName in zip(imgNames,labelNames):
        imgPath = imgsDir+'/'+imgName
        labelPath = labelsDir+'/'+labelName

        img = np.array([cv2.imread(imgPath)])
        labelVec = dataHandler.loadNPArr(labelPath)
        label = dataHandler.labelVecToImg(labelVec)

        pred = model.predict(img)[0]
        '''
        b,g,r = cv2.split(pred)

        cv2.imshow('Img : ',img[0])
        cv2.waitKey(0)

        cv2.imshow('Prediction : ',pred)
        cv2.waitKey(0)

        cv2.imshow('b : ',b)
        cv2.waitKey(0)

        x = np.zeros((pred.shape[0],pred.shape[1]))
        x[b>np.average(b)] = 1
        cv2.imshow('b > avg : ',x)
        cv2.waitKey(0)
        '''
        predVec = dataHandler.tensorToPrediction(pred)
        acc = dataHandler.compareImgs(labelVec,predVec)
        accuracies.append(acc)
        print(acc)

        if(showImgs):
            cv2.imshow('Prediction : ',pred)
            cv2.waitKey(0)
            '''
            predImg = dataHandler.labelVecToImg(predVec, 'RGB')
            cv2.imshow('Predicted RGB: ',predImg)
            cv2.waitKey(0)
            cv2.imshow('Actual: ',label)
            cv2.waitKey(0)
            '''
        i += 1
        print('Validation in Progress: %i/%i\t%0.2f%%'%(
            i,
            len(imgNames),
            100*(i)/len(imgNames)
        ))
    return(sum(accuracies)/len(accuracies))

if __name__ == "__main__":
    samples = 10
    sampleIndexes = np.random.randint(0,len(imgNames),size=(samples))
    sampleImgNames = [imgNames[i] for i in sampleIndexes]
    sampleLabelNames = [labelNames[i] for i in sampleIndexes]

    acc = validate(ImgsDir, LabelsDir, sampleImgNames, sampleLabelNames, loadedModel, showImgs=True)
    print('Model Accuracy: %0.3f'%(acc))