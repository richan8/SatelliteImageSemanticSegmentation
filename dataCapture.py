#### Imports
import pyautogui
import numpy as np
import time
from PIL import Image

#### Global Vars
outputDir = 'data/raw'

count = 0
def capture():
  global count
  raw = np.asarray(pyautogui.screenshot())
  
  img = raw[80:-60,8:948,:]
  label = raw[80:-60,955:1895,:]
  
  Image.fromarray(img).save(outputDir+'/imgs/'+'img'+str(count)+'.png')
  Image.fromarray(label).save(outputDir+'/labels/'+'label'+str(count)+'.png')
  print(count)
  count += 1

def runCapture():
  time.sleep(5)
  while(True):
    capture()
    time.sleep(0.3)

runCapture()