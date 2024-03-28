import numpy as np
import os
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,Model

from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten

from sklearn.model_selection import train_test_split 

path='Class 74/Face-image-dataset-small'


'''lable of the file 0,1,2'''


detector=FaceDetector(minDetectionCon=0.8)

dataSet_length=len(os.listdir(path))
print(dataSet_length)

ages=[]
images=[]

for img in os.listdir(path):
    try:
        # print(img)
        if img!='.git':
            age=img.split("_")[0]
            # print(age)
            # Read the image and store in RGB format
            image=cv2.imread(str(path)+"/"+str(img))
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image,bbox=detector.findFaces(image,draw=True)
            # print(bbox)
            if bbox:
                X,Y,W,H=bbox[0]["bbox"]
                #crop the face out of the image
                croppedImage=image[Y:Y+H,X:X+H]

                #resizedImage
                resizedImage=cv2.resize(croppedImage,(200,200))
            images.append(resizedImage)
            ages.append(age)

            #Display the age on the image
            image=cv2.putText(image,"Age: "+age,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            # cv2.imshow("image",image)
            # cv2.waitKey(0)
             
    except:
        print("Error while Reading Image")

ages=np.array(ages,dtype=np.int64)
images=np.array(images)
# print(type(ages))

# print("Age: ",ages)
# print(images)





#Machine_Learning Model Creation using neural
training_images,testing_images,training_ages,testing_ages=train_test_split(images,ages)
print("length of training images: ",len(training_images))
print("length of testing images: ",len(testing_images))
print("length of training ages: ",len(training_ages))
print("length of testing ages: ",len(testing_ages))
print(".................................................")
age_model=Sequential()
age_model.add(Conv2D(128,kernel_size=3,activation='relu',input_shape=(200,200,3)))
age_model.add(MaxPool2D(pool_size=3,strides=2))

age_model.add(Conv2D(128,kernel_size=3,activation='relu'))
age_model.add(MaxPool2D(pool_size=3,strides=2))

age_model.add(Conv2D(256,kernel_size=3,activation='relu'))
age_model.add(MaxPool2D(pool_size=3,strides=2))

age_model.add(Conv2D(512,kernel_size=3,activation='relu'))
age_model.add(MaxPool2D(pool_size=3,strides=2))

#classification
#2D to 1D ARRAY
age_model.add(Flatten())

#drop out layer
age_model.add(Dropout(0.2))

#dense layer
age_model.add(Dense(512,activation='relu'))

age_model.add(Dense(1,activation='linear', name="age"))

#compiling model

age_model.compile(optimizer="adam",loss="mae")
# print(age_model.summary())

#train model
history=age_model.fit(training_images,training_ages,validation_data=(testing_images,testing_ages),epochs=10)

# print(history)
# age_model.save("age_model.h5")


loss=history.history['loss']
epochs=range(1,len(loss)+1)

val_loss=history.history['val_loss']

plt.plot(epochs,loss,'y',label="training loss")
plt.plot(epochs,val_loss,'y',label="Validation loss")

plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()