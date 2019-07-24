# face_recog.py

import face_recognition
import cv2
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = cv2.VideoCapture(0)

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        # self.body = []

    def __del__(self):
        del self.camera

    def compare(self, a):
        global tempsize
        if tempsize == []:
            tempsize.append(a)
            print(tempsize[0],"off")
        elif tempsize != []:
            if tempsize[0] < (a):
                print("저장값",tempsize)
                print(a)
                print("대상이 가까워졌습니다")
                del tempsize[0]
                tempsize.append(a)

            elif tempsize[0] > a:
                print(tempsize)
                print("대상이 멀어졌습니다")
                del tempsize[0]
                tempsize.append(a)
        return tempsize

    '''def savedistance(self, temp):
        self.tempsize = []
        return tempsize'''
    def get_frame(self):
        global tempsize
        hog = cv2.HOGDescriptor()

        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # cv2.startWindowThread()

        # Grab a single frame of video
        (grabbed, frame) = self.camera.read()

        # Resize frame of video to 1/4 size for faster face recognition processing

        # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = frame[:, :, ::-1]


        # Only process every other frame of video to save time
        if self.process_this_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # self.boxes, weights = hog.detectMultiScale(gray, winStride=(8,8), scale=1.05)
            self.boxes, weights = hog.detectMultiScale(gray)
            self.boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in self.boxes])


            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # print(type(body))
        # for (x, y, w, h) in self.body:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)


        for (xA, yA, xB, yB) in self.boxes:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # Display the results



        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):

            a=right-left
            tempsize=self.compare(a)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            #print((left + right)/2, (top + bottom)/2)
            #print((right - left), (bottom - top))

        return frame



    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    tempsize = []
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
