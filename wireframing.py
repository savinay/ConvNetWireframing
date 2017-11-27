
# coding: utf-8

# In[22]:

from matplotlib import pyplot
from PIL import Image
from pylab import *
import os
import numpy as np
import cv2


# In[3]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train


# In[67]:

path1 = 'dataset/image/'
path2 = 'dataset/video/'
vector=[]
y_train = []
for image in os.listdir(path1):
    try:
        im = cv2.imread(path1 + image)
        im = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )
        resized = cv2.resize(im, (100,100), interpolation = cv2.INTER_AREA)
        vector.append(resized)
        y_train.append(0)
    except:
        continue
for image in os.listdir(path2):
    try:
        im = cv2.imread(path2 + image)
        im = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )
        resized = cv2.resize(im, (100,100), interpolation = cv2.INTER_AREA)
        vector.append(resized)
        y_train.append(1)
    except:
        continue
print vector


# In[66]:

for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    
    pyplot.imshow(np.asarray(vector[i]), cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()


# In[74]:

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')
vector = np.asarray(vector).reshape(np.asarray(vector).shape[0],1,100,100)
X_train = vector
X_train = X_train.astype('float32')
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    # create a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(100, 100), cmap=pyplot.get_cmap('gray'))
    # show the plot
    pyplot.show()
    break


# In[76]:

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# fit parameters from data
datagen.fit(X_train)
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(100, 100), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break


# In[2]:

import os
import shutil
path1 = 'dataset/image/'
path2 = 'dataset/video/'

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
        
print image_list
from sklearn.model_selection import train_test_split
train, test = train_test_split(image_list, test_size=0.2, random_state=14)
print len(train)
print len(test)

# unique_train = []
# unique_train = set([x.split(".")[0] for x in train])
# print len(unique_train)
# unique_test = set([x.split(".")[0] for x in test])
# print len(unique_test)


# In[5]:

from keras.preprocessing.image import ImageDataGenerator

path1 = 'dataset/image'
path2 = 'dataset/video'
img_class = os.listdir(path1)
vid_class = os.listdir(path2)


for image in train:
    if image in img_class:
        full_file_name = os.path.join(path1, image)
        shutil.copy(full_file_name, "dataset/train/image/" + image)
    elif image in vid_class:
        full_file_name = os.path.join(path2, image)
        shutil.copy(full_file_name, "dataset/train/video/" + image)
        
for image in test:
    if image in img_class:
        full_file_name = os.path.join(path1, image)
        shutil.copy(full_file_name, "dataset/test/image/" + image)
    elif image in vid_class:
        full_file_name = os.path.join(path2, image)
        shutil.copy(full_file_name, "dataset/test/video/" + image)
# #     if not os.path.exists("dataset/" + image):
# #         os.makedirs("dataset/train/" + label)
# #     full_file_name = os.path.join(path, image)
# #     shutil.copy(full_file_name, "data/train/" + label)
# #     os.rename("data/train/" + label + "/" + image, "data/train/" + label + "/" + image + ".png")

# for image in test:
#     label = image.split(".")[0]
#     if not os.path.exists("data/test/" + label):
#         os.makedirs("data/test/" + label)
#     full_file_name = os.path.join(path, image)
#     shutil.copy(full_file_name, "data/test/" + label)
#     os.rename("data/test/" + label + "/" + image, "data/test/" + label + "/" + image + ".png")


# In[17]:

from keras import backend as K
img_width = 150
img_height = 150
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
batch_size = 16
train_data_dir = 'dataset/train'
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')  # since we use binary_crossentropy loss, we need binary labels

test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')


# In[18]:

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


# In[19]:

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[20]:

model.fit_generator(
        train_generator,
        steps_per_epoch=len(train)//batch_size,
        epochs=50,
        validation_data=test_generator,
       validation_steps=len(train)//batch_size)
model.save_weights('first_try.h5') 
model.save('my_model.h5')


# In[33]:

img = cv2.imread('dataset/video/fail_test.png')
img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
# im = array(Image.open('dataset/image/image1.png'))
print img.shape
resized = cv2.resize(img, (150,150), interpolation = cv2.INTER_AREA)
# resized = np.array(im)
print resized.shape
resized = np.reshape(resized, ( 150, 150, 1))
img = np.expand_dims(resized, axis=0)
print img.shape

# print model.predict_classes(resized, verbose=0)
# cv2.resize(im, (80,61), interpolation = cv2.INTER_AREA)

model.predict(img, batch_size=16, verbose=0)


# In[ ]:



