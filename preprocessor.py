#### Imports
#import pyautogui
import numpy as np
from PIL import Image
from PIL import ImageFilter #PIL.Image.filter(ImageFilter.MinFilter(3))
from PIL import ImageEnhance
import os
import cv2
import random

#### Global Vars
inputImgsDir = 'data/raw/imgs'
inputLabelsDir = 'data/raw/labels'
outputImgsDir = 'data/final/imgs'
outputLabelsDir = 'data/final/labels'

imgFormats = ['jpeg','png','jpeg']
imgNames = [x for x in sorted(os.listdir(inputImgsDir)) if x.split('.')[-1] in imgFormats]
labelNames = [x for x in sorted(os.listdir(inputLabelsDir)) if x.split('.')[-1] in imgFormats]

expectedDim = (464,464)

def processLabel(imgDir,labelDir):
    ## Images are in BGR
    im = cv2.imread(labelDir)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ### Buildings
    buildingMask = np.logical_and(im[:,:,2]>100, im[:,:,2]<250)
    buildings = np.zeros([im.shape[0],im.shape[1]],dtype=np.uint8)
    buildings.fill(0)
    buildings[buildingMask] = 255

    buildings = Image.fromarray(buildings)
    # Removes Noise
    buildings = buildings.filter(ImageFilter.MinFilter(3))
    buildings = buildings.filter(ImageFilter.MaxFilter(5))
    buildings = np.array(buildings)

    # Creating the final mask
    buildingMask = buildings>0

    ### Roads
    roadMask = im[:,:,2]<100
    roads = np.zeros([im.shape[0],im.shape[1]],dtype=np.uint8)
    roads.fill(0)
    roads[roadMask] = 255

    roads = Image.fromarray(roads)
    ## Removes any really thin roads
    ## This thin roads are most often invisible paths that go through car parks etc.
    ## It is best we avoid these
    roads = roads.filter(ImageFilter.MinFilter(7))
    roads = roads.filter(ImageFilter.MaxFilter(11))
    roads = np.array(roads)

    # Creating the final mask
    roadMask = roads>0

    ### VEGETATION
    img = cv2.imread(imgDir)
    PILImg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = np.array(ImageEnhance.Color(PILImg).enhance(2))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Thresholding HSV values to get pixels with vegetation
    # Took alot of time to perfect but finally it works great
    sensitivity = 30
    midH = 50
    startS = 50
    endS = 255
    startV = 0
    endV = 255
    mask = cv2.inRange(hsv, (midH-sensitivity, startS, startV), (midH+sensitivity, endS, endV))
    
    # Smoothing
    mask = np.array(Image.fromarray(mask).filter(ImageFilter.MinFilter(5)))
    mask = np.array(Image.fromarray(mask).filter(ImageFilter.MaxFilter(7)))
    
    # Creating the final mask
    vegetationMask = mask>0

    ### Combining all masks into one image
    res = np.zeros([im.shape[0],im.shape[1]],dtype=np.int16)
    res.fill(0)
    res[roadMask] = 84
    res[buildingMask] = 255
    res[vegetationMask] = 132

    # Largest Filter applied is size 11 which means we need a minimum 5 pixels of padding.
    # Therefore the label and image are cropped by 5 px
    # We add an additional crop of 1 pixel to make the size 464x464 (Divisible by 2^5 for our model)
    res = res[5:res.shape[0]-6,5:res.shape[1]-6]

    return(res)

def processImg(imgDir):
    ## IMAGES ARE READ IN BGR
    im = cv2.imread(imgDir)

    # Filter size 11 applied in label which means we need a 5 pixel padding.
    # Therefore the label and image are cropped by 5 px
    # We add an additional crop of 1 pixel to make the size 464x464 (Divisible by 2^5 for our model)
    im = im[5:im.shape[0]-6,5:im.shape[1]-6,:]
    return(im)

### Augumentation 
# It's optimized for square images since they are already available to us.

# Returns flips of all images in the list
def flips(imgs):
    res = []
    for img in imgs:
        res.extend([
            img,
            cv2.flip(img, flipCode=0),
            cv2.flip(img, flipCode=1),
            cv2.flip(img, flipCode=-1)
        ])
    return(res)

# Returns n random rotations of a center slice
def rotations(img, rotationList):
    m = int((img.shape[0])/4)
    n = int((img.shape[0])/2)

    res = []
    for rot in rotationList:
        res.append(np.array(Image.fromarray(img).rotate(rot))[m:m+n,m:m+n])

    return(res)

# Returns quarter size slices
def slices(img):
    n = int((img.shape[0])/2)
    return([
        img[0:n,0:n],
        img[0:n,n+1:],
        img[n+1:,0:n],
        img[n+1:,n+1:]
    ])

# Applies all above augumentations to Image 
def augument(img, rotationList):
    res = []
    res.extend(slices(img))
    res.extend(rotations(img, rotationList))
    res.extend(flips(res))
    
    # Ensuring the shape of Inputs and Outputs are as expected
    # Global Var 'expectedDim' can be set to None to skip check
    global expectedDim
    if(expectedDim is not None):
        unexpectedShapes = [x.shape for x in res if x.shape[0]!=464 or x.shape[1]!=464]
        if(len(unexpectedShapes)>0):
            print('Error - Unexpected Shapes Found: ',unexpectedShapes)
            exit()
    
    return(res)

# Safety check before running the Augumentation
if(len(imgNames) != len(labelNames) or len(imgNames) == 0):
    print('Image/label length mismatch')

pairIndex = 0
for imgName,labelName in zip(imgNames,labelNames):
    imgDir = inputImgsDir+'/'+imgName
    labelDir = inputLabelsDir+'/'+labelName

    img = processImg(imgDir)
    label = processLabel(imgDir, labelDir)

    # Generating random rotation list to augument labels and images.
    numRotations = 8
    rotationList = [random.randint(0,360) for _ in range(numRotations)]

    for i,img in enumerate(augument(img, rotationList)):
        cv2.imwrite(outputImgsDir+'/'+'img-%i-%i.jpeg'%(pairIndex, i), img)
    
    for i,label in enumerate(augument(label, rotationList)):
        cv2.imwrite(outputLabelsDir+'/'+'label-%i-%i.jpeg'%(pairIndex, i), label)

    print('Augumentation in Progress: %i/%i\t%0.2f%%'%(
        pairIndex+1,
        len(imgNames),
        100*(pairIndex+1)/len(imgNames)
    ))
    pairIndex += 1