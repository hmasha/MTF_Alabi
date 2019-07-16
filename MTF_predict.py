#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:08:32 2019
https://stackoverflow.com/questions/18869550/python-pil-cut-off-my-16-bit-grayscale-image-at-8-bit
https://stackoverflow.com/questions/8832714/how-to-use-multiple-wildcards-in-python-file-dialog
https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

@author: hillary.masha
"""
# Helper libraries
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import Model
import cv2
import wx

#fileselector for test image

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

IMG_Width = 30
IMG_Height = 270

#resize and store test data in array
try:
    image = cv2.imread (filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_Width, IMG_Height))
    test_data.append (image)
except Exception as e:
    print(str(e))

test_data = np.array(test_data)
test_data = test_data.astype('float32') / 255

print(test_data)

test_X = test_data.reshape(-1,IMG_Height,IMG_Width,1)

# load model
model = load_model("model.h5")
print("Loaded model from disk")

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print("\nPredicted Class")
print(predicted_classes)


#display Result
if predicted_classes[0] == 1:
    print("PASS")
    arr = np.asarray(test_X.reshape(IMG_Height,IMG_Width))
    plt.imshow(arr, cmap='gray')
    plt.title("Prediction for {} : PASS".format(imgname))
    plt.show()
        
elif predicted_classes[0] == 0:
    print("FAIL")
    arr = np.asarray(test_X.reshape(IMG_Height,IMG_Width))
    plt.imshow(arr, cmap='gray')
    plt.title("Prediction for {} : FAIL".format(imgname))
    plt.show()




        



