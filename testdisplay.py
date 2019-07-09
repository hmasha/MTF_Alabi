#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:08:32 2019

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
import os

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

def get_path(wildcard):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path
filename = get_path('*.png')

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

correct = np.where(predicted_classes==test_labels)[0]
incorrect = np.where(predicted_classes!=test_labels)[0]
print(correct)
print(incorrect)


#display Result
if predicted_classes[0] == 1:
    print("PASS")
    for i, correct in enumerate(correct[:0]):
        plt.figure(figsize=[10,10])
        plt.subplot(121)
        plt.imshow(test_X[correct].reshape(IMG_Height,IMG_Width), cmap='gray', interpolation='none')
        plt.title("PASS: Predicted {}, Actual {}".format(predicted_classes[correct], test_labels[correct]))
        plt.tight_layout()

else:
    print("FAIL")
    for i, incorrect in enumerate(incorrect[:0]):
        plt.figure(figsize=[10,10])
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[incorrect].reshape(IMG_Height,IMG_Width), cmap='gray', interpolation='none')
        plt.title("FAIL: Predicted {}, Actual {}".format(predicted_classes[incorrect], test_labels[incorrect]))
        plt.tight_layout()

plt.figure(figsize=[10,10])
plt.imshow(test_X[0].reshape(IMG_Height,IMG_Width), cmap='gray', interpolation='none')
plt.title("FAIL: Predicted {}, Actual {}".format(predicted_classes[0], test_labels[0]))
plt.show




