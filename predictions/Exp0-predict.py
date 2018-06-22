## Predict phase with the last checkpoint 
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
from keras import backend as K
import tensorflow as tf


_TRAIN_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/train/"
_TRAIN_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/train/"
_DEV_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/dev/"
_DEV_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/dev/"
_IMG_INDEX_ = "/home/2/18M31438/jobs/GoogleLandmarkJobs/imgList/x_train.csv"
_TEST_IMG_INDEX_ = "/home/2/18M31438/GContest2/code/test.csv"
_IMG_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/imgs/'
_OUT_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/'
_TRAIN_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/train/'
_DEV_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/dev/'
_VAL_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/val/'
_TEST_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/test/imgs/'
_FEATURES_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/features/'
_CHECKPOINTS_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/checkpoints/exp1/'

batch_size = 64
n_classes = 14951
m = 264520
m_t = 66131
m_v = 117673

           

# Checkpoint format and log directory.
checkpoint_format = _CHECKPOINTS_ +'checkpoint-{epoch}.h5'
log_dir = './logs'


# Enough epochs to demonstrate learning rate warmup and the reduction of
# learning rate when training plateaues.
epochs = 20

# Input image dimensions
img_rows, img_cols = 224,224

learning_rate = 0.001

def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[0] for line in csvreader]
    return key_url_list


## Get minibatch
# Validation data iterator.
val_gen = ImageDataGenerator(rescale=1./255, preprocessing_function=keras.applications.resnet50.preprocess_input)
val_iter = test_gen.flow_from_directory(_VAL_DIR_, batch_size=batch_size,
                                         target_size=(224, 224))

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


## Get the earliest checkpoint
resume_from_epoch = 0
for try_epoch in range(epochs, 0, -1):
    if os.path.exists(checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.4,
            horizontal_flip=True,
            preprocessing_function=keras.applications.resnet50.preprocess_input)



train_iter = train_gen.flow_from_directory(_TRAIN_0_DIR_, batch_size=batch_size,
                                           target_size=(224, 224))

##Creating the model
ResNet50_test = FineTuneModel((224,224,3))
#print("model created")

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0:
    ResNet50_test.load_weights(checkpoint_format.format(epoch=resume_from_epoch))
    print("checkpoint loaded %d" % resume_from_epoch)

opt = keras.optimizers.adam(lr=learning_rate)

#Adding a compiler ! :D "mean_squared_error"
ResNet50_test.compile(optimizer=opt, loss = keras.losses.categorical_crossentropy, metrics = ['accuracy', 'top_k_categorical_accuracy'])

steps = m_v // batch_size
val_gen = val_minibatch_gen()


imgs_names = ParseData(_TEST_IMG_INDEX_)
#print(len(imgs_names))

print(train_iter.class_indices)
file = open("predictions.txt", "w")
file.write("id,landmarks\n")

dic = {}

##Preparing class dictionary

print("start")
for cat, ind in train_iter.class_indices.items():
    dic[ind] = cat

print(dic)

for i in range(steps):
    #print("step %d, \n" % i)
    imgs, names = next(val_gen)
    predictions = ResNet50_test.predict(imgs)
    y_classes = predictions.argmax(axis=-1)
    print(y_classes)
    predictionsIndex = [(np.argmax(x), x[np.argmax(x)])  for x in predictions]
    #print(predictionsIndex)
    
    
    for j, pred in enumerate(predictionsIndex):
        #print(str(names[j].decode("utf-8")))
        imgs_names.remove(str(names[j].decode("utf-8") ))
        file.write(str(names[j].decode("utf-8") ) + "," + str(y_classes[j]) + ' ' + "{0:.2f}".format(pred[1]) + "\n")
        

for name in imgs_names:
    file.write(str(name) + ",\n")
    
file.close()
    

    
    
    
