
from __future__ import print_function
import numpy as np
import os, glob, csv
import h5py

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf


_CHECKPOINTS_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/checkpoints/exp1/'
_EXP_VAL_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/val/'
_TEST_IMG_INDEX_ = "/gs/hs0/tga-nlp-titech/erick/data/index/test.csv"


# Checkpoint format and log directory.
checkpoint_format = _CHECKPOINTS_ +'checkpoint-{epoch}.h5'
log_dir = './logs'


# dimensions of our images
img_width, img_height = 224, 224
epochs = 20
batch_size = 64
learning_rate = 0.001

def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[0] for line in csvreader]
    return key_url_list

def FineTuneModel(input_shape=(224,224,3), num_classes = 6155):
    
    X_input = Input(input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=X_input)
    X = Flatten()(base_model.output)
    X = Dense(4096, activation='sigmoid')(X)
    X = Dropout(0.2)(X)
    X = Dense(4096, activation='sigmoid')(X)
    X = Dropout(0.1)(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(num_classes, activation='softmax')(X)
    
    model = Model(inputs = X_input, outputs = X, name='fineTuneModel')    
    return model


##Creating the model
ResNet50_test = FineTuneModel((224,224,3))

resume_from_epoch = 0
for try_epoch in range(epochs, 0, -1):
    if os.path.exists(checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break
        
if resume_from_epoch > 0:
    ResNet50_test.load_weights(checkpoint_format.format(epoch=resume_from_epoch))
    print("checkpoint loaded %d" % resume_from_epoch)
    
opt = keras.optimizers.adam(lr=learning_rate)

#Adding a compiler ! :D "mean_squared_error"
ResNet50_test.compile(optimizer=opt, loss = keras.losses.categorical_crossentropy, metrics = ['accuracy', 'top_k_categorical_accuracy'])
    
val_dir = _EXP_VAL_DIR_
val_list = sorted(glob.glob(val_dir + "*"))
names = [os.path.splitext(os.path.basename(x))[0] for x in file_list]

imgs_names = ParseData(_TEST_IMG_INDEX_)

batch = []
file = open("predictions.txt", "w")
file.write("id,landmarks\n")

for name in names[:10]:
    img = image.load_img(val_dir + name + ".jpg", target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    prediction = ResNet50_test.predict_on_batch(x)
    y_classes = predictions.argmax(axis=-1)
    
    imgs_names.remove(name)
    file.write(name + "," + str(y_classes[0]) + ' ' + "{0:.2f}".format(prediction[0][1]) + "\n")
               
for name in imgs_names:
    file.write(str(name) + ",\n")
    
file.close()
    

