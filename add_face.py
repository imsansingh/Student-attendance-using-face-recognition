import cv2
import pickle
import numpy as np
import os

# using webcam to capture images
video=cv2.VideoCapture(0)

# to capture face features
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")

# to keep camera on for infinite time
while True:
    # to read the images
    ret,frame=video.read()

    # to convert image into black and white
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to cover-up the structure of the face
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)

    # cropping and resizing it
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50))

        # collecting 50 images
        if len(faces_data)<=50 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        # to put image count
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        # to make box around the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    # to stop the camera : when we press q or after taking 30 images camera will stop 
    if k==ord('q') or len(faces_data)==50:
        break
video.release()
cv2.destroyAllWindows()

# converting the images data into numoy array and reshaping it
faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(50, -1)

# for first time it create a names.pkl file and store names in it 
if 'names.pkl' not in os.listdir('data/'):
    names=[name]*50
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# if names.pkl is already existing, just append further names to it
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*50
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# for first time it create a faces.pkl file and store names in it 
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
