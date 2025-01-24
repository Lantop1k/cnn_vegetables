import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D, Activation, Embedding

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

import os
import cv2


train_folder='Vegetable Images/train'    # training folder

vegtrain=os.listdir(train_folder)        # sub folders in the training folders (vegetable classes)

trainy=[]                                # list to store labels

class_samples =800                       # Number of sub group for each data class

trainSize=class_samples*len(vegtrain)     # Total number of training samples

c=0

# Extracting image array for training 
trainimagesX=np.zeros((trainSize,64,64))

for veg in vegtrain:  # loop through the subfolders (vegetable classes)
    
    vs=os.path.join(train_folder,veg) # extract full path for a vegetable class
    
    files=os.listdir(vs)              # list all files for the vegetable class
    
    counter=0  # set counter 
    
    for f in files:  # loop through all the files
        
        file=os.path.join(os.path.join(train_folder,veg),f) #get the full path
        
        img=cv2.imread(file,0) # extract the image array with openCV
         
        img=cv2.resize(img,(64,64)) # reshape the image as 64 by 64
        
        trainimagesX[c,:,:]=img     # store the image array
        trainy.append(vegtrain.index(veg))
        
        c+=1        # increase counter 
        counter+=1  # increase counter 
        
        if counter>=class_samples:  # break after the number of samples has been extracted 
            break


val_folder='Vegetable Images/validation'    # validation folder

vegval=os.listdir(val_folder)        # sub folders in the testing folders (vegetable classes)

testy=[]                                # list to store labels

class_samples =200                       # Number of sub group for each data class

testSize=class_samples*len(vegval)     # Total number of testing samples

c=0 

# Extracting image array for testing
testimagesY=np.zeros((testSize,64,64))
imgs=[]

for veg in vegval:  # loop through the subfolders (vegetable classes)
    
    vs=os.path.join(val_folder,veg) # extract full path for a vegetable class
    
    files=os.listdir(vs)              # list all files for the vegetable class
    
    counter=0  # set counter 
    
    for f in files:  # loop through all the files
        
        file=os.path.join(os.path.join(val_folder,veg),f) #get the full path
        
        img=cv2.imread(file,0) # extract the image array with openCV
        img_org=cv2.imread(file)
         
        img=cv2.resize(img,(64,64)) # reshape the image as 64 by 64
        
        testimagesY[c,:,:]=img     # store the image array
        testy.append(vegval.index(veg))
        
        c+=1        # increase counter 
        counter+=1  # increase counter 
        
        if counter>=class_samples:  # break after the number of samples has been extracted
            imgs.append(img_org)
            break
        

plt.figure(figsize=(10,8))
for i in range(len(vegval)):
    plt.subplot(3,5,i+1)
    plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
    plt.title(vegval[i])
    plt.axis('off')
plt.show()



trainy=np.reshape(trainy, (len(trainy),1))
testy=np.reshape(testy, (len(testy),1))

print(testimagesY.shape, testy.shape)

trainimagesX = np.reshape(trainimagesX/255, (trainimagesX.shape[0],64,64,1))
testimagesY = np.reshape(testimagesY/255, (testimagesY.shape[0],64,64,1))


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(64, 64, 1)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
L=len(vegtrain)
model.add(Dense(L, activation='softmax'))

# compiling the sequential model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


model.fit(trainimagesX, trainy, batch_size=32, epochs=50, validation_data=(testimagesY, testy))


ypred=list(model.predict_classes(testimagesY))

ytest_str= [vegtrain[i] for i in ypred]
y= [vegtrain[int(i)] for i in testy]

print('========== Classification Report =================')
print(classification_report(ytest_str,y))

print('========== Confusion Matrix ===============')
print(confusion_matrix(ytest_str,y))
