#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:08:32 2019
https://stackoverflow.com/questions/18869550/python-pil-cut-off-my-16-bit-grayscale-image-at-8-bit
https://stackoverflow.com/questions/8832714/how-to-use-multiple-wildcards-in-python-file-dialog
https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

@author: hillary.masha
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import flask
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
from keras.utils import to_categorical
from keras.models import load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import scipy
import keras
from keras.models import model_from_json
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import cv2
import glob
import numpy as np

#fileselector for test image
import wx

#get user input, file 


def get_path( ):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Select Test Image(.tif,.png,.jpg)', wildcard="pictures (*.jpeg,*.png,*.tif,*.jpg)|*.jpeg;*.png;*.tif;*.jpg", style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
        picfile = dialog.GetFilename()
        for item in wx.GetTopLevelWindows():
            if isinstance(item, wx.Dialog):
                item.Destroy()
            item.Close()
    else:
        path = None
    dialog.Destroy()
    return path, picfile
filename , imgname = get_path()

y_test = []
test_data = []
""" files7 = glob.glob ("/Users/hillary.masha/testMNIST/fail/*.png")
files8 = glob.glob ("/Users/hillary.masha/testMNIST/pass/*.png") """

IMG_Width = 30
IMG_Height = 270

#resize and store test data in array
try:
    image = cv2.imread (filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_Width, IMG_Height))
    test_data.append (image)
    y_test.append(0)
except Exception as e:
    print(str(e))


y_test = np.array(y_test)
test_data = np.array(test_data)
test_data = test_data.astype('float32') / 255


print(test_data.shape)
print(y_test.shape)


test_images = test_data
test_labels = y_test
class_names = ['Pass', 'Fail']

test_X = test_images.reshape(-1,IMG_Height,IMG_Width,1)

# Change the labels from categorical to one-hot encoding
test_Y_one_hot = to_categorical(test_labels)

batch_size = 64
epochs = 70
num_classes = 2

# load model
model = load_model("model.h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

""" test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1]) """

predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print("\nPredicted Class")
print(predicted_classes)

correct = np.where(predicted_classes==test_labels)[0]
incorrect = np.where(predicted_classes!=test_labels)[0]

#display Result
if predicted_classes[0] == 1:
    print("PASS")
    arr = np.asarray(test_X.reshape(IMG_Height,IMG_Width))
    plt.imshow(arr, cmap='gray')
    plt.title("Prediction for {} : PASS".format(imgname))
    plt.show()
        
else:
    print("FAIL")
    arr = np.asarray(test_X.reshape(IMG_Height,IMG_Width))
    plt.imshow(arr, cmap='gray')
    plt.title("Prediction for {} : FAIL".format(imgname))
    plt.show()




        



