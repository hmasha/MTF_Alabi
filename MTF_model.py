#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:20:59 2019
https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

@author: hillary.masha
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
#from tensorflow import keras
from keras.utils import to_categorical

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import keras
from keras.models import load_model
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import cv2
import glob
import numpy as np
train_data = []
y_train = []
y_test = []
test_data = []
#list of training folders
files = glob.glob ("/Users/hillary.masha/Pass/*.png")
files2 = glob.glob ("/Users/hillary.masha/Pass/*.jpg")
files3 = glob.glob ("/Users/hillary.masha/Pass/*.tif")
files4 = glob.glob ("/Users/hillary.masha/Fail/*.png")
files5 = glob.glob ("/Users/hillary.masha/Fail/*.jpg")
files6 = glob.glob ("/Users/hillary.masha/Fail/*.tif")
#list of test folders
files7 = glob.glob ("/Users/hillary.masha/testMNIST/fail/*.png")
files8 = glob.glob ("/Users/hillary.masha/testMNIST/pass/*.png")
#files8 = glob.glob ("/Users/hillary.masha/Pass/*.tif")
#files7 = glob.glob ("/Users/hillary.masha/Fail/*.tif")
IMG_Width = 30
IMG_Height = 270

for myFile in files:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(1)
    except Exception as e:
        print(str(e))
for myFile in files2:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(1)
    except Exception as e:
        print(str(e))
for myFile in files3:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(1)
    except Exception as e:
        print(str(e))
for myFile in files4:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(0)
    except Exception as e:
        print(str(e))
for myFile in files5:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(0)
    except Exception as e:
        print(str(e))
for myFile in files6:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        train_data.append (image)
        y_train.append(0)
    except Exception as e:
        print(str(e))

print("\nReading Test Images")
for myFile in files7:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        test_data.append (image)
        y_test.append(0)
    except Exception as e:
        print(str(e))
for myFile in files8:
    print(myFile)
    try:
        image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_Width, IMG_Height))
        test_data.append (image)
        y_test.append(1)
    except Exception as e:
        print(str(e))

train_data = np.array(train_data)
train_data = train_data.astype('float32') / 255
y_train = np.array(y_train)


y_test = np.array(y_test)
test_data = np.array(test_data)
test_data = test_data.astype('float32') / 255

print(train_data.shape)
print(y_train.shape)
print(test_data.shape)
print(y_test.shape)


train_images = train_data
test_images = test_data
train_labels = y_train
test_labels = y_test
class_names = ['Pass', 'Fail']

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])

# =============================================================================
# # Display the first image in training data
# plt.subplot(121)
# plt.imshow(train_images[0,:,:], cmap='gray')
# plt.title("Ground Truth : {}".format(train_labels[0]))
#
# # Display the first image in testing data
# plt.subplot(122)
# plt.imshow(test_images[0,:,:], cmap='gray')
# plt.title("Ground Truth : {}".format(test_labels[0]))
# =============================================================================

train_X = train_images.reshape(-1,IMG_Height,IMG_Width, 1)
test_X = test_images.reshape(-1,IMG_Height,IMG_Width,1)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label:', train_labels[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.02, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

batch_size = 64
epochs = 200
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMG_Height,IMG_Width,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

model.summary()
model.save("model.h5")

model = load_model("model.h5")
print("Loaded model from disk")

test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
#plt.figure()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print('\nTest loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Prediction
predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

num_test = len(test_X)
print("\nNumber of test images" , num_test)


plt.figure(figsize=[10,10])
#Store number correctly identified images
correct = np.where(predicted_classes==test_labels)[0]
print ("\nFound %d correct labels" % len(correct))
for i, correct in enumerate(correct[:num_test]):
    plt.subplot(4,4,i+1)
    plt.imshow(test_X[correct].reshape(IMG_Height,IMG_Width), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Actual {}".format(predicted_classes[correct], test_labels[correct]))
    #plt.tight_layout()

incorrect = np.where(predicted_classes!=test_labels)[0]
print ("\nFound %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:num_test]):
    plt.subplot(4,4,i+1)
    plt.imshow(test_X[incorrect].reshape(IMG_Height,IMG_Width), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    #plt.tight_layout()

print("\nClass 0 = Fail , Class 1 = Pass")
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))









