"""
==================================================
|        Train Model  (Practical Script)         |
==================================================
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# NOTE `onlyfiles` contains list of files of frontal face sample images.
"""
> [f for f in listdir(data_path) if isfile(join(data_path,f))]: list the sample image files within "./faces/" directory.
    * listdir(data_path)        - show list of files in "./faces/" directory.
    * isfile(join(data_path,f)) - return True when file of "./faces/user<count>.jpg" do exist.
        Therefore, the (latter) variable f takes a single file from list from `./faces/`,
        and if it really exist (checked by isfile() function) the file is placed as an element of the list via (former) variable f.
"""
data_path = './faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# Create empty lists to include training data and its lable for ML.
Training_Data, Labels = [], []

# SECTION Fill in training data and its label with optimal format.
for i, files in enumerate(onlyfiles):       # enumerate() is a function that lists the elements(`files`) in the list with sequencing number(`i`).
    image_path = data_path + onlyfiles[i]   # set full relative path into single variable `image_path`.
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)      # Read the image data in greyscale format from the relative path.
    Training_Data.append(np.asarray(images, dtype=np.uint8))   # Include the image data into the Training Data as unsigned 8-bit integer.
    Labels.append(i)                        # sequencing number is used as a label.

# Converts the number to 32-bit integer
Labels = np.asarray(Labels, dtype=np.int32)

# NOTE Trains model from OpenCV API, using Local Binary Pattern Histogram algorithm.
"""
> LBPHFaceRecongnizer_create() is inherited from FaceRecognizer() which is inherited from Algorithm() which could work as one of the framework for facial ML.
    * The function only accepts greyscale images.
    * train() method is an attribute inherited from FaceRecognizer().
    * In C++, it is LBPHFaceRecognizer::create().
"""
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# SECTION Function`detector`: pass the argument to `img` and detect the frontal face. 
def face_detector(img, size = 0.5):

    """
    > `cv2.cvtColor`: converts RGB format `img` to greyscale
    > `detectMultiScale()`: pass grayscaled `img` and detect frontal face based on classifier from a file; returns [x,y,w,h].
        * scaleFactor = 1.3 - how much the image is reduced at each image scale.
        * minNeighbors = 5  - how many neighbors each candidate rectangle should have to retain it to eliminate False detection; refer to kNN.
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)          # although does not convert actual video screenshot, but maybe used for faster computation?
    faces = face_classifier.detectMultiScale(gray,1.3,5) # reduce the scale by `img`/1.3 pixel and detected as True when there's five neighbors.

    if faces is ():
        return img,[]

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

# SECTION Loop for activation and usage of function `face_extractor` to extract facial sample.
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() # `ret` is boolean True when capture is successful in spite of facial detection. Frame contains screenshot image data.

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)   # Returns predicted label and its distnace (i.e., cost).

        # Converts cost to prediction confidence...presumably not based on algorithm but intuitively.
        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        # If the confidence (probability) is greater than 75 unlocks the device (if there is one).
        if confidence > 75:
            # visualize the sample counter under the window name "Face Cropper" showing "face" image.
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

        # If the confidence (probability) is greater lesser 75 locks the device (if there is one).
        else:
            # visualize the sample counter under the window name "Face Cropper" showing "face" image.
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    # But when no face is detected...
    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    # End the video capturing function when pressed Enter button.
    if cv2.waitKey(1)==13:
        break

cap.release()              # Close the video capture device.
cv2.destroyAllWindows()    # Destroys all of the opened HighGUI windows.
