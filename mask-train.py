import os
import numpy as np
#importing tensorflow modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

INIT_LR = 1e-4
BS=32
EPOCHS=20

print("loading images...")
imgpath = dataset
data=[]
labels=[]

#looping the dataset
for dirs in os.listdir(imgpath):
    label=os.path.join(imgpath,dirs)
    if not os.path.isdir(label):
        continue
    
    for item in os.listdir(label):
        if item.startswith("."):
            continue

        image = load_img(os.path.join(label, item), target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        #append the list
        data.append(image)
        labels.append(label)    

data = np.array(data, dtype="floatt32")
labels = np.array(labels) #converting data and labels to numpy arrays

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#partitioning into training and testing(75% train 25%test)

(trainX,testX,trainY,testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

#load mobilennetv2 nw
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

#constructing the headmodel
headModel= baseModel.output
headModel=AveragePooling2D(pool_size=(7, 7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(128, activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2, activation="softmax")(headModel)

model= Model(inputs=baseModel.input, outputs=headModel)
 
for layer in baseModel.layers:
    layer.trainable= False

print("compiling model..")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#training headmdel
print("training..")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

predindex=model.predict(testX, batch_size=BS)
predindex=np.argmax(predindex, axis=1)
print(classification_report(testY.argmax(axis=1), predindex, target_names=lb.classes_))

#save the model
print("saving mask detector model...")
model.save("MaskDetector.h5")