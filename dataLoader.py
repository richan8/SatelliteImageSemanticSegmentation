import numpy as np
from tensorflow.keras.utils import Sequence
import dataHandler

class DataLoader(Sequence):
    def __init__(self, imgNames, labelNames, imgDir, labelDir, batchSize, nClasses, dim, toFit=True, imgChannels=3, shuffle=True):
        self.imgNames = imgNames
        self.labelNames = labelNames
        self.imgDir = imgDir
        self.labelDir = labelDir
        self.toFit = toFit
        self.batchSize = batchSize
        self.dim = dim
        self.imgChannels = imgChannels
        self.nClasses = nClasses
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Returns number of batches per epoch
        return(int(np.floor(len(self.imgNames) / self.batchSize)))

    def __getitem__(self, batchIndex):
        # Generate one batch of data for the given batchIndex
        indexes = self.indexes[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize]
        batchImgNames = [self.imgNames[idx] for idx in indexes]
        if self.toFit:
            batchLabelNames = [self.labelNames[idx] for idx in indexes]
            return(self.genImgs(batchImgNames), self.genLabels(batchLabelNames))
        else:
            return(self.genImgs(batchImgNames))

    def on_epoch_end(self):
        # Shuffle indexes every epoch
        self.indexes = np.arange(len(self.imgNames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def genImgs(self, imgNames):
        imgs = np.zeros((self.batchSize, self.dim[0], self.dim[1], self.imgChannels), dtype=np.uint8)
        for i, imgName in enumerate(imgNames):
            imgs[i] = dataHandler.loadImg(self.imgDir+'/'+imgName)
        return(imgs)

    def genLabels(self, labelNames):
        labels = np.zeros((self.batchSize, self.dim[0], self.dim[1], self.nClasses), dtype=np.uint8)
        for i, labelName in enumerate(labelNames):
            labels[i] = dataHandler.loadNPArr(self.labelDir+'/'+labelName)
        return(labels)