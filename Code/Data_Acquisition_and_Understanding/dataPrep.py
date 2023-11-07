# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
import pandas as pd
import os
import zipfile

# Loading Data
currentfolder = os.getcwd()
train_data_df = pd.read_csv(currentfolder + '/training.csv')
test_data_df = pd.read_csv(currentfolder + '/test.csv')

# Describing Data
train_data_df.head()
test_data_df.head()
pd.Series(train_data_df.columns)
pd.Series(test_data_df.columns)
print(train_data_df.info())
print(test_data_df.info())
for col in train_data_df.columns.tolist():          
    print('{} column missing values: {}'.format(col, train_data_df[col].isnull().sum()))
train_data_df[train_data_df['left_eye_center_x'].isnull()]
train_data_df[train_data_df['left_eye_inner_corner_x'].isnull()]

# Splitting data to train, test, and labels
def process_data(data_df,train):
    if train:
        y=np.array(data_df.iloc[:,:30])

    img_dt = []
    for i in range(len(data_df)):
        img_dt.append(data_df['Image'][i].split(' '))

    X=np.array(img_dt, dtype='float')
    return X,y if train else " "
X_train,y_train=process_data(train_data_df,True)
X_test_submit,_=process_data(test_data_df,False)
print(X_train)
print(y_train)
print(X_test_submit)

# Fixing null values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

for i in range(30):
    imputer.fit(y_train[:,i].reshape(-1,1))
    y_train[:,i]=(imputer.transform(y_train[:,i].reshape(-1,1))).reshape(-1,)
print(y_train[556,0])
print(y_train[2239,0])
print(y_train[1600,4])
print(y_train[1654,4])
print(X_train.shape)
print(y_train.shape)
print(X_test_submit.shape)
X_train=X_train.reshape(X_train.shape[0],96,96)
X_test_submit=X_test_submit.reshape(X_test_submit.shape[0],96,96)

# Exploring Data
plt.imshow(X_train[0], cmap='gray')
plt.title("Input Image")
plt.savefig('plot.png', bbox_inches='tight')
plt.show()
plt.imshow(X_test_submit[0], cmap='gray')
plt.title("Input Image")
plt.savefig('plot.png', bbox_inches='tight')
plt.show()
plt.imshow(X_train[0], cmap='gray')
plt.scatter(y_train[0][0::2], y_train[0][1::2], c='red', marker='o')
plt.title("Image with Facial Keypoints")
plt.show()
X_train=X_train/255.0
X_test_submit=X_test_submit/255.0
Final_X_train=X_train
Final_y_train=y_train