import numpy as np

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
            (0,0,0) : 0,
            (1,0,0) : 84,
            (0,1,0) : 255,
            (0,0,1) : 132,
        },
        'RGB':{
            'dim'   : 3,
            (0,0,0) : [255,255,255],
            (1,0,0) : [100, 120, 180],
            (0,1,0) : [220, 100, 140],
            (0,0,1) : [100, 180, 120],
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
    Given that we have switched to uncompressed labels, threshold for label comparision is set to 0.
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

def saveNPArr(img, imgDir):
    with open(imgDir, 'wb') as f:
        np.save(f, np.array(img), allow_pickle=True)

def loadNPArr(imgDir):
    with open(imgDir, 'rb') as f:
        return(np.load(f, allow_pickle=True))

def tensorToPrediction(labelPred, thresold = 0.4):
    for i,row in enumerate(labelPred):
        for j,x in enumerate(row):
            temp = np.zeros((3),dtype=np.uint8)
            if(np.max(x)>thresold):
                temp[np.argmax(x)] = 1
            labelPred[i,j]=temp
    return(labelPred)