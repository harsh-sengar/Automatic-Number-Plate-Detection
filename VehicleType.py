
# coding: utf-8

# In[8]:


import cv2
from PIL import Image
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
# from keras.optimizers import RMSprop
from keras import optimizers
size= (32, 32)


# In[9]:


def GetModel():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation = "relu", kernel_initializer = 'he_normal' , padding = "same", input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2),  strides = (1,1)))
    model.add(Conv2D(128, (5, 5), activation = "relu", padding = "same", kernel_initializer = 'he_normal'))
    # model.add(MaxPooling2D(pool_size = (2, 2),  strides = (1,1)))
    model.add(Flatten())
    model.add(Dense(256, activation = "relu",  kernel_initializer = 'he_normal'))
    # model.add(Dense(128, activation = "relu",  kernel_initializer = 'he_normal'))
    # model.add(Dense(256, activation = "relu",  kernel_initializer = 'he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation = "softmax"))
    sgd = optimizers.SGD(lr = 0.0008)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def IdentifyVehicle(image):
    model = GetModel()
    model.load_weights('C:\\Users\\Archit\\NeuralNetwork workshop\\Minor2Final\\Identification\\VehicleModel.h5')
    OriginamImage = cv2.imread(image)
    OriginamImage = cv2.resize(OriginamImage, size)
    imgGray = cv2.cvtColor(OriginamImage, cv2.COLOR_BGR2GRAY)
    imgGray = imgGray.reshape((1, 32, 32, 1))
    ypred = model.predict(imgGray)
    if ypred[0][0] == 0:
        return 'bus'
    elif ypred[0][0] == 1:
        return 'car'
    else:
        return 'pickup_truck'


# image = 'C:\\Users\\Archit\\Desktop\\frame2.jpg'
# IdentifyVehicle(image)

