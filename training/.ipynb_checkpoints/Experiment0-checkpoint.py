#### fine tuned model with horovod, for few shot learning 

from __future__ import print_function
import numpy as np
import os, glob, csv
import h5py
import math
import socket

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import horovod.keras as hvd



_TRAIN_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/train/"
_TRAIN_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/train/"
_DEV_0_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp0/dev/"
_DEV_1_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/exp1/dev/"
_VAL_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/val/"
_INDEX_DIR_ = "/gs/hs0/tga-nlp-titech/erick/data/index/"
_CHECKPOINTS_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/checkpoints/exp1/'

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

print("Machine: %s, Process ID: %s, rank %d" % (socket.gethostname(), str(os.getpid()), hvd.rank()))

##Load dataset function Finally!!!! :D

_IMG_INDEX_ = "/home/2/18M31438/jobs/GoogleLandmarkJobs/imgList/x_train.csv"
_IMG_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/imgs/'
_OUT_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/'
_TRAIN_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/train/'
_DEV_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/dev/'
_VAL_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/datasets/val/'
_TEST_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/test/imgs'
_FEATURES_DIR_ = '/gs/hs0/tga-nlp-titech/erick/data/features/'



## Parameters 
batch_size = 64
n_classes = 14951
m = 264520
m_t = 66131

steps = m // batch_size
test_steps = m_t // batch_size

# Checkpoint format and log directory.
checkpoint_format = _CHECKPOINTS_ +'checkpoint-{epoch}.h5'
log_dir = './logs'


epochs = 20

# Input image dimensions
img_rows, img_cols = 224,224
learning_rate = 0.001

def FineTuneModel(input_shape=(224,224,3), num_classes = 14951):
    
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


# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.adam(lr=learning_rate * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)


callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=2, min_lr=0.00001, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(_CHECKPOINTS_+'checkpoint-{epoch}.h5'))
    
    
## Get the earliest checkpoint
resume_from_epoch = 0
for try_epoch in range(epochs, 0, -1):
    if os.path.exists(checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break
        
print(resume_from_epoch)
# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')



# Training data iterator.
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

# Validation data iterator.
test_gen = image.ImageDataGenerator(
    zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
test_iter = test_gen.flow_from_directory(_DEV_0_DIR_, batch_size=batch_size,
                                         target_size=(224, 224))


##Creating the model
ResNet50_test = FineTuneModel((224,224,3))
#print("model created")

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    ResNet50_test.load_weights(checkpoint_format.format(epoch=resume_from_epoch))
    print("checkpoint loaded %d" % resume_from_epoch)


# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Set up ImageDataGenerators to do data augmentation for the training images.
#train_gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
#                               height_shift_range=0.08, zoom_range=0.08)
#test_gen = ImageDataGenerator()


#Adding a compiler ! :D "mean_squared_error"
ResNet50_test.compile(optimizer=opt, loss = keras.losses.categorical_crossentropy, metrics = ['accuracy', 'top_k_categorical_accuracy'])

#Running the model for testing for 20 epochs
ResNet50_test.fit_generator(train_minibatch_gen(mini_batch_size = batch_size),
                  callbacks=callbacks,
                  steps_per_epoch=steps // hvd.size(), 
                  epochs = epochs, 
                  verbose = 1, 
                  initial_epoch=resume_from_epoch)



#ResNet50_test.fit_generator(train_minibatch_gen(),
#                  steps_per_epoch=train_batches // hvd.size(),
#                  callbacks=callbacks,
#                  epochs = epochs, 
#                  verbose = 1, 
#                  initial_epoch=resume_from_epoch,
#                  validation_data=test_gen.flow(X_test, Y_test, batch_size=batch_size),
#                  validation_steps=3 * test_batches // hvd.size())

score = ResNet50_test.evaluate_generator(test_minibatch_gen(mini_batch_size = batch_size),
                                         steps = test_steps // hvd.size(),
                                         verbose=1)


print('Test loss:', score[0])
print('Test accuracy:', score[1])