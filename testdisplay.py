#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:08:32 2019
https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/?fbclid=IwAR04aMA6iR8U1hhN6UiqDZOCLFUS7maG0SvJ1A0N6EzoLibJvLVNDpEF1-A
@author: hillary.masha
"""
#Helper libraries
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

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

#img_path = '/Users/hillary.masha/Raw/In.jpg'
image = cv2.imread(filename)

# load the image, clone it, and setup the mouse callback function
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key=cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

# if there are two reference points, then crop the region of interest
# from the image display and save it
if len(refPt) == 2:
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imwrite("ROI.png", roi)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

# close all open windows
cv2.destroyAllWindows()

#store cropped image in numpy array
cropped_img = "/Users/hillary.masha/code/MTF_Alabi/ROI.png"

y_test = []
test_data = []

IMG_Width = 30
IMG_Height = 270

#resize and store test data in array
try:
    image = cv2.imread (cropped_img, cv2.IMREAD_GRAYSCALE)
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

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
    plt.title("Prediction for image : PASS")
    plt.show()
        
else:
    print("FAIL")
    arr = np.asarray(test_X.reshape(IMG_Height,IMG_Width))
    plt.imshow(arr, cmap='gray')
    plt.title("Prediction for image : FAIL")
    plt.show()
















