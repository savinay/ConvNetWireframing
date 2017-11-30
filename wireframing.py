
# coding: utf-8

# In[5]:

from matplotlib import pyplot
# from keras.datasets import mnist
from PIL import Image
from pylab import *
import os
import numpy as np
import cv2
import scipy.misc as image
import tensorflow as tf


# In[89]:

import keras


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[8]:

import os
import shutil
path1 = 'dataset/image/'
path2 = 'dataset/video/'
path3 = 'dataset/navbar'
path4 = 'dataset/button'
path5 = 'dataset/dropdown'

image_list = []
for image in os.listdir(path1):
    try:
        if image != "Icon\r" and image != ".DS_Store":
            image_list.append(image)
    except:
        continue
        
for image in os.listdir(path2):
    try:
        if image != "Icon\r" and image != ".DS_Store":
            image_list.append(image)
    except:
        continue
        
for image in os.listdir(path3):
    try:
        if image != "Icon\r" and image != ".DS_Store":
            image_list.append(image)
    except:
        continue
        
for image in os.listdir(path4):
    try:
        if image != "Icon\r" and image != ".DS_Store":
            image_list.append(image)
    except:
        continue
        
for image in os.listdir(path5):
    try:
        if image != "Icon\r" and image != ".DS_Store":
            image_list.append(image)
    except:
        continue
        
print image_list
from sklearn.model_selection import train_test_split
train, test = train_test_split(image_list, test_size=0.2, random_state=14)
print len(train)
print len(test)



# In[ ]:




# In[18]:

from keras.preprocessing.image import ImageDataGenerator

path1 = 'dataset/image'
path2 = 'dataset/video'
path3 = 'dataset/navbar'
path4 = 'dataset/button'
path5 = 'dataset/dropdown'

img_class = os.listdir(path1)
vid_class = os.listdir(path2)
navbar_class = os.listdir(path3)
button_class = os.listdir(path4)
dropdown_class = os.listdir(path5)



for image in train:
    if image in img_class:
        full_file_name = os.path.join(path1, image)
        shutil.copy(full_file_name, "dataset/train/image/" + image)
    elif image in vid_class:
        full_file_name = os.path.join(path2, image)
        shutil.copy(full_file_name, "dataset/train/video/" + image)
    elif image in navbar_class:
        full_file_name = os.path.join(path3, image)
        shutil.copy(full_file_name, "dataset/train/navbar/" + image)
    elif image in button_class:
        full_file_name = os.path.join(path4, image)
        shutil.copy(full_file_name, "dataset/train/button/" + image)
    elif image in dropdown_class:
        full_file_name = os.path.join(path5, image)
        shutil.copy(full_file_name, "dataset/train/dropdown/" + image)
        
for image in test:
    if image in img_class:
        full_file_name = os.path.join(path1, image)
        shutil.copy(full_file_name, "dataset/test/image/" + image)
    elif image in vid_class:
        full_file_name = os.path.join(path2, image)
        shutil.copy(full_file_name, "dataset/test/video/" + image)
    elif image in navbar_class:
        full_file_name = os.path.join(path3, image)
        shutil.copy(full_file_name, "dataset/test/navbar/" + image)
    elif image in button_class:
        full_file_name = os.path.join(path4, image)
        shutil.copy(full_file_name, "dataset/test/button/" + image)
    elif image in dropdown_class:
        full_file_name = os.path.join(path5, image)
        shutil.copy(full_file_name, "dataset/test/dropdown/" + image)


# In[67]:

import scipy.misc as image

X = []
Y = []
kernel = np.ones((3,4),np.uint8)

path1 = 'dataset/image/'
for img in os.listdir(path1):
    try:
        im = image.imread('dataset/image/' + img)
        im = image.imresize(im , (150, 150))
        X.append(im)
        Y.append(0)
    except Exception as e:
        continue
        
path2 = 'dataset/video/'
for img in os.listdir(path2):
    try:
        im = image.imread('dataset/video/' + img)
        im = image.imresize(im , (150, 150))
        X.append(im)
        Y.append(1)
    except Exception as e:
        continue
        
path3 = 'dataset/navbar/'
for img in os.listdir(path3):
    try:
        im = image.imread('dataset/navbar/' + img)
        im = image.imresize(im , (150, 150))
        X.append(im)
        Y.append(2)
    except Exception as e:
        continue
        
path4 = 'dataset/button/'
for img in os.listdir(path4):
    try:
        im = image.imread('dataset/button/' + img)
        im = image.imresize(im , (150, 150))
        X.append(im)
        Y.append(3)
    except Exception as e:
        continue


path5 = 'dataset/dropdown/'
for img in os.listdir(path5):
    try:
        im = image.imread('dataset/dropdown/' + img)
        im = image.imresize(im , (150, 150))
        
        X.append(im)
        Y.append(4)
    except Exception as e:
        continue

X = np.asarray(X)
Y = np.asarray(Y)
print len(X)
print len(Y)

plt.imshow(X[0])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=14)

print len(y_train)
print len(y_test)


# In[55]:

X_train = X_train.reshape(X_train.shape[0], 150, 150, 4)
X_test = X_test.reshape(X_test.shape[0], 150, 150, 4)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255
print X_train.shape
print y_train.shape


# In[56]:

import keras
from keras import utils as np_utils
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
print y_train.shape
num_classes = 5
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 4) 


# In[57]:

from keras import backend as K
print K.image_data_format()
img_width = 150
img_height = 150
if K.image_data_format() == 'channels_first':
    input_shape = (4, img_width, img_height)
else:
    input_shape = (img_width, img_height, 4)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[58]:

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_gen = ImageDataGenerator()


# In[59]:

train_generator = datagen.flow(X_train, y_train, batch_size=16)
test_generator = test_gen.flow(X_test, y_test, batch_size=16)


# In[61]:

model.fit_generator(train_generator, steps_per_epoch=len(X_train)//16, epochs=100, validation_data=test_generator, validation_steps = len(X_test)//16)


# In[52]:

plot_history(history)


# In[86]:

model.save('latest_model.h5')


# In[80]:

def make_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model







#Cross Validation


seed = 7
np.random.seed(seed)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cvscores = []
for train, test in kfold.split(X, Y):
    model = make_model()
    print len(X[train])
    print len(Y[test])
    print Y[train].shape
    print Y[train]
    tr = Y[train].reshape(Y[train].shape[0], 1)
    te = Y[test].reshape(Y[test].shape[0], 1)
    print Y[train].shape
    num_classes = 5
    Y_train = np_utils.to_categorical(tr, num_classes)
    Y_test = np_utils.to_categorical(te, num_classes)
    train_generator = datagen.flow(X[train], Y_train, batch_size=16)
    test_generator = test_gen.flow(X[test], Y_test, batch_size=16)
    model.fit_generator(train_generator, steps_per_epoch=len(X[train])//16, epochs=100, validation_data=test_generator, validation_steps = len(X[test])//16)
    scores = model.evaluate(X[test], Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)


# In[85]:

print "Cross Validation Accuracy: ", ("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[ ]:

# Fully Connected Neural Network Model



# In[21]:




# In[22]:




# In[23]:




# In[ ]:




# In[40]:

img = cv2.imread('dataset/image/image1.png')
img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
resized = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
print resized.shape
resized = np.reshape(resized, ( 150, 150, 1))
img = np.expand_dims(resized, axis=0)
print img.shape


model.predict(img, batch_size=16, verbose=0)


# In[50]:




# In[ ]:




# In[87]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[88]:




# In[ ]:




# In[ ]:



