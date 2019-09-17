"""
==================================================
|        Train Model (Conceptual Script)         |
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

# SECTION 
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
