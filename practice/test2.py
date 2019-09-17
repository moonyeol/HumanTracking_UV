import numpy as np
import cv2
from matplotlib import pyplot as plt
import camera

cam = camera.VideoCamera()

body_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_fullbody.xml')
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# cv2.startWindowThread()
# open webcam video stream
# the output will be written to output.avi

while (True):
    frame = cam.get_frame()
    # resizing for faster detection
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # rgb_small_frame = frame[:, :, ::-1]

    # detect people in the image
    # returns the bounding boxes for the detected objects
    faces_in_body = body_cascade.detectMultiScale(gray, 1.01, 10)
    for (xf,yf,wf,hf) in faces_in_body:
        cv2.rectangle(frame,(xf,yf),(xf+wf,yf+hf),(0,255,0),2)

    # Write the output video

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# finally, close the window
