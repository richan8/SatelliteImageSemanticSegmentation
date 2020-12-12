import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import cv2
import os

def labelVecToImg(labelVec, mode = 'G'):
    '''
    Converts labelVector to image
    Modes:
        'G'     : Greyscale compatible with input labels
        'RGB'   : RGB to be visually nice
    '''
    dMap = {
        'G':{
            'dim'   : 1,
            (0,0,0,1) : 30,
            (1,0,0,0) : 84,
            (0,1,0,0) : 255,
            (0,0,1,0) : 132,
        },
        'RGB':{
            'dim'   : 3,
            (0,0,0,1) : [40,40,40],
            (1,0,0,0) : [100, 120, 180],
            (0,1,0,0) : [220, 100, 140],
            (0,0,1,0) : [100, 180, 120],
        }
    }

    if(mode not in dMap):
        print('Incorrect mode %s in labelToVec function.\nPlease select modes from the following:'%(mode), end = ' ')
        print(', '.join([x for x in dMap.keys() if x != 'dim']))
        exit()

    img = np.zeros((labelVec.shape[0],labelVec.shape[1],dMap[mode]['dim']),dtype=np.uint8)
    for i,row in enumerate(labelVec):
        for j,x in enumerate(row):
            img[i,j] = dMap[mode][tuple(x)]
    
    if(img.shape[-1]==1):
        img = np.squeeze(img)
    return(img)

def compareImgs(img1,img2,threshold = 0):
    '''
    Compares 2 images considering the minor errors in intensities in image compression.
    Returns the fraction of pixels that match given the threshold of error.
    Given that we have switched to uncompressed labels, threshold for label comparision is set to 0 so we check exact match.
    '''
    if(img1.shape != img2.shape):
        print('Incompatible image shapes during image comparision: ',img1.shape,img2.shape)
        exit()

    n = 0
    c = 0

    imgShape = img1.shape
    isGrey = len(imgShape) == 2

    for i,row in enumerate(img1):
        for j,_ in enumerate(row):
            n += 1
            if(isGrey): # Greyscale Image
                if(abs(int(img1[i,j])-int(img2[i,j])) <= threshold):
                    c += 1
            elif([abs(int(img1[i,j,x])-int(img2[i,j,x])) <= threshold for x in range(imgShape[2])].count(True) == imgShape[2]): # N-Dimensional Image
                c += 1
    return(c/n)

# load the image fileNames for the entire dataset.
def getImgNames(inputImgsDir, imgFormats):
    return(np.array([x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]))

# load the label fileNames for the entire dataset.
def getLabelNames(inputLabelsDir, labelFormats):
    return(np.array([x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in labelFormats]))

def saveImg(img, imgPath):
    cv2.imwrite(imgPath, img)

def loadImg(imgPath):
    return(cv2.imread(imgPath))

def saveNPArr(npArr, npArrPath):
    with open(npArrPath, 'wb') as f:
        np.save(f, np.array(npArr), allow_pickle=True)

def loadNPArr(imgPath):
    with open(imgPath, 'rb') as f:
        return(np.load(f, allow_pickle=True))

# Convert one hot encoded output to predicion vector
def tensorToPrediction(labelPred, img, thresold = 0.4):
    # Getting Vegetation mask from the image
    PILImg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = np.array(ImageEnhance.Color(PILImg).enhance(2))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sensitivity = 35
    midH = 50
    startS = 50
    endS = 255
    startV = 0
    endV = 255
    mask = cv2.inRange(hsv, (midH-sensitivity, startS, startV), (midH+sensitivity, endS, endV))
    mask = np.array(Image.fromarray(mask).filter(ImageFilter.MinFilter(5)))
    mask = np.array(Image.fromarray(mask).filter(ImageFilter.MaxFilter(7)))
    vegetationMask = mask>0
    for i,row in enumerate(labelPred):
        for j,x in enumerate(row):
            temp = np.zeros((4),dtype=np.uint8)
            temp[np.argmax(x)] = 1
            labelPred[i,j] = temp
    labelPred[vegetationMask] = (0,0,1,0)
    return(labelPred)