import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 3

if __name__ == "__main__":

    # sample img
    gray = []
    im = np.random.randint(256, size=(5,940,940,3))
    im_type = np.array(im, dtype=np.uint8)
    for i in range(im.shape[0]):
        gray.append(cv2.cvtColor(im_type[i], cv2.COLOR_RGB2GRAY))
    gray_img = np.array(gray)
    print(gray_img.shape)

    # thresholding (binary for now)
    #thresholding = [threshold_multiotsu(img, classes=4) for img in gray_img]
'''
    # Hyperparameters for our network
    input_size = 5*940*940*3
    hidden_sizes = [128, 64]
    output_size = 940*940*3
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.Softmax(dim=1))
    print(model)
    model
    '''

    # tensorflow u-net image segmentation

