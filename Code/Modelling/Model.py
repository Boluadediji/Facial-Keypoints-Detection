# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
import pandas as pd
import os
import zipfile

currentfolder = os.getcwd()
train_data_df = pd.read_csv(currentfolder + '/training.csv')
test_data_df = pd.read_csv(currentfolder + '/test.csv')

# Splitting training data to train and test to evaluate the models
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=1)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Function to prepare different models
def model_arch(arch):
    model=(arch((96, 96, 1)))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_dataset,epochs=100,validation_data=test_dataset,batch_size=64,shuffle=True)
    return model

# Architecture 6
def arch6(input_shape):

    input_img = tf.keras.Input(shape=input_shape)
    
    layer=tfl.Conv2D(filters= 38 , kernel_size= 5,strides=(2, 2),padding='same')(input_img)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    layer=tfl.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

    layer=tfl.DepthwiseConv2D(kernel_size= 3 ,strides=(2, 2),padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    
    layer=tfl.Conv2D(filters= 196 , kernel_size= 3,strides=(2, 2),padding='same')(layer)
    layer=tfl.BatchNormalization(axis=3)(layer,training=True)
    layer=tfl.ReLU()(layer)
    
    layer=tfl.Flatten()(layer)

    layer=tfl.Dense(units=500, activation='relu')(layer)
    layer=tfl.Dropout(0.2)(layer)

    outputs=tfl.Dense(units= 30 , activation='linear')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
model6=model_arch(arch6)

# Downloading the model
import zipfile
import os
from IPython.display import FileLink

def zip_dir(directory = os.curdir, file_name = 'model.zip'):
    """
    zip all the files in a directory

    Parameters
    _____
    directory: str
        directory needs to be zipped, defualt is current working directory

    file_name: str
        the name of the zipped file (including .zip), default is 'directory.zip'

    Returns
    _____
    Creates a hyperlink, which can be used to download the zip file)
    """
    os.chdir(directory)
    zip_ref = zipfile.ZipFile(file_name, mode='w')
    for folder, _, files in os.walk(directory):
        for file in files:
            if file_name in file:
                pass
            else:
                zip_ref.write(os.path.join(folder, file))

    return FileLink(file_name)
zip_dir()

train_dataset = tf.data.Dataset.from_tensor_slices((Final_X_train, Final_y_train)).batch(64)

# train one of the best models on all training data
model6_Final=(arch6((96, 96, 1)))
model6_Final.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['accuracy'])
model6_Final.summary()
history = model6_Final.fit(train_dataset,epochs=150,batch_size=64,shuffle=True)
