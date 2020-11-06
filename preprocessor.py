#### Imports
import pyautogui
import numpy as np
from PIL import Image
from PIL import ImageFilter #PIL.Image.filter(ImageFilter.MinFilter(3))
from PIL import ImageEnhance
import os
import cv2

#### Global Vars
inputImgsDir = 'data/raw/imgs'
inputLabelsDir = 'data/raw/labels'

outputDir = 'data/final'

imgNames = os.listdir(inputImgsDir)
labelNames = os.listdir(inputLabelsDir)

im = Image.open(inputLabelsDir+'/'+labelNames[0])
im = np.array(im)

### BUILDINGS
buildingMask = np.logical_and(im[:,:,2]>100, im[:,:,2]<250)
buildings = np.zeros([im.shape[0],im.shape[1]],dtype=np.uint8)
buildings.fill(255)
buildings[buildingMask] = 128

buildings = Image.fromarray(buildings)
buildings = buildings.filter(ImageFilter.MaxFilter(3))
buildings = buildings.filter(ImageFilter.MinFilter(5))
buildings = np.array(buildings)

### ROADS
roadMask = im[:,:,2]<100
roads = np.zeros([im.shape[0],im.shape[1]],dtype=np.uint8)
roads.fill(255)
roads[roadMask] = 0

roads = Image.fromarray(roads)
roads = roads.filter(ImageFilter.MinFilter(3))
roads = np.array(roads)

### VEGETATION
## USING
'''
rawImg = Image.open(inputImgsDir+'/'+imgNames[0])
rawImg = ImageEnhance.Color(rawImg).enhance(3)
rawImg.show()
rawImg = np.array(rawImg)

vegetation = np.zeros([im.shape[0],im.shape[1]],dtype=np.int16)
#vegetation = (2*rawImg[:,:,1])/(rawImg[:,:,0]+rawImg[:,:,2])

print(vegetation)
print(np.min(vegetation))
print(np.max(vegetation))

vegetation = Image.fromarray(vegetation*128)
vegetation.show()
'''

'''
## USING CV2 HSVs
im = cv2.imread(inputImgsDir+'/'+imgNames[0])
cv2.imshow('image',im)
cv2.waitKey(0)

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV',hsv)
cv2.waitKey(0)

#mask = cv2.inRange(hsv, (36, 0, 0), (86, 255,255))
#imgMask = mask>0

#vegetation = np.zeros([img.shape[0],img.shape[1]],dtype=np.uint8)
#vegetation[imgMask] = 255
#cv2.imshow('test',vegetation)
#cv2.waitKey(0)

vegetationMask = np.logical_and(hsv[:,:,0]>95, hsv[:,:,0]<175)
vegetationMask = np.logical_and(vegetationMask, hsv[:,:,1]>50)
vegetationMask = np.logical_and(vegetationMask, hsv[:,:,2]>20)
vegetation = np.zeros([im.shape[0],im.shape[1]],dtype=np.uint8)
vegetation[vegetationMask] = 255
cv2.imshow('Vegetation',vegetation)
cv2.waitKey(0)

'''
### COMBINING THE 2
res = np.zeros([im.shape[0],im.shape[1]],dtype=np.int16)
res.fill(255)
res[buildings==128] = 128
res[roads==0] = 0
#res[vegetation] = 64
#print(res.shape)
#res = Image.fromarray(res)
#res.show()
#print(res)
cv2.imwrite('test.png', res)