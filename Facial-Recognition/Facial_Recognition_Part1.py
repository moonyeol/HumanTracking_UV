"""
==================================================
|             Extract Training Data              |
==================================================
"""

import cv2
import numpy as np

# NOTE `cv2.CascadeClassifier`: load a classifier from a file "./haarcascade_frontalface_default.xml" which only detects frontal face.
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


# SECTION Function`face_extractor`: pass the argument to `img` and detect the frontal face. 
def face_extractor(img):

    """
    > `cv2.cvtColor`: converts RGB format `img` to greyscale
    > `detectMultiScale()`: pass grayscaled `img` and detect frontal face based on classifier from a file; returns [x,y,w,h].
        * scaleFactor = 1.3 - how much the image is reduced at each image scale.
        * minNeighbors = 5  - how many neighbors each candidate rectangle should have to retain it to eliminate False detection; refer to kNN.
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          # although does not convert actual video screenshot, but maybe used for faster computation?
    faces = face_classifier.detectMultiScale(gray,1.3,5) # reduce the scale by `img`/1.3 pixel and detected as True when there's five neighbors.

    # if none face is detected returns nothing, ending the function early.
    # else when face is detected, proceed to next function statement.
    if faces is ():
        return None

    # Assign data from faces literally to variable of x, y, w, h and use it crop the face out from `img` using numpy.
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    # return cropped face from `img`.
    return cropped_face


# SECTION Loop for activation and usage of function `face_extractor` to extract facial sample.
cap = cv2.VideoCapture(0)   # Activate video capturing device such as webcam.
count = 0                   # Counts how many screenshot of the video was captured.

while True:
    ret, frame = cap.read() # `ret` is boolean True when capture is successful in spite of facial detection. Frame contains screenshot image data.

    # 
    if face_extractor(frame) is not None:
        count+=1   # increase counter as face is detected.

        # normalize the frontal face samples and greyscale them.
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # save the frontal face sample under the following path.
        file_name_path = './faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        # visualize the sample counter under the window name "Face Cropper" showing "face" image.
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    
    # If no face detected via returned value of "None"
    else:
        print("Face not Found")
        pass

    # "cv2.waitKey(1)": returns the pressed key after specified delayed ms time; if 0ms means wait for a key event infinitely.
    # End the screenshot when acquired 100 images or pressed Enter button.
    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()              # Close the video capture device.
cv2.destroyAllWindows()    # Destroys all of the opened HighGUI windows.
print('Colleting Samples Complete!!!')
