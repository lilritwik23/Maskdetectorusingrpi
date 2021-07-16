
#import required modules
import os
import cv2
import time

from tensorflow.python.ops.math_ops import truediv
import imutils
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

def detect_mask(frame, faceNet, maskNet):
    (h,w)=frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    faceNEt.setInput(blob)
    detections=facenet.forward()

    #making list of faces and their predictions and locaiton
    faces=[]
    predictions=[]
    locations=[]

    for i in range(0, detections.shape[2]):
        conf=detections[0,0,i,2] #extracting the confidence
        if conf>0.5:
            #make a box around the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X1, Y1, X2, Y2) = box.astype("int")
            (X1,Y1)=(max(0,X1),max(0,Y1))
            (X2,Y2)=(min(w-1,X2),min(h-1,Y2))

            #processing the ROI
            face=frame[X1:X2,Y1:Y2]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB) #CHANGE TO rgb COLORING
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            locations.append((X1,Y1,X2,Y2))

    if len(faces)>0: #prediction list will only be appended if there is atleast one face
        faces = np.array(faces, dtype="float32")
        predictions=maskNet.predict(faces, batch_size=32)

    return (locations,predictions)

prototxtPath = "face_detector/deploy.prototxt"
weightsPath =  "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#initializing
print('loading model...')
maskNet = load_model("maskdetector.h5")

print('Initializing camera...')
vs=VideoStream(src=0).start()
time.sleep(2.0)
#looping the frames of the vs
while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=500)
    #passing the values through the created function
    (locations,predicitons)=detect_mask(frame, faceNet,maskNet)
    for (box,predicitons) in zip(locations,predicitons):
        (X1,Y1,X2,Y2)=box
        (mask, withoutMask)=predicitons
          
        #labelling
        if mask > withoutMask:
            label="Mask on. Thank you"
            color=(0,255,0)
        else:
            label="no mask detected"
            color=(255,0,0)
           
        
        cv2.putText(frame, label, (startX-50, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        #output frame
    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()








