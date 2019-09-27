#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/dnn/dict.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>    /* Standard input/output definitions */
#include <string>
#include <string.h>   /* String function definitions */
#include "Recognizer.hpp"





    Recognizer::Recognizer(String landmarkDat, String frNetModel, string odConfigFile, string odWeightFile, String fdConfigFile, String fdWeightFile, String classesFile){
        // ASSIGN VARIABLE "net" AS AN OBJECT OF "anet_type" DEFINED ABOVE.
        /*
            >> `models/dlib_face_recognition_resnet_model_v1.dat`: DNN for a "FACIAL RECOGNITION".
                ...it is presume this file too needs deserialization which reconstructs to original data format.
                ...for more information of a serialization, read this webpage; https://www.geeksforgeeks.org/serialization-in-java/.
            >> statement `dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;`
                ...reconstructed recognition model is placed in a hollow neural network frame manually created using operator `>>`.
                ...where now this variable `net` works as a model.
        */
        deserialize(frNetModel) >> faceEncoder;
        deserialize(landmarkDat) >> pose_model;
        // CAFFE의 설정파일과 모델파일을 OpenCV에서 미리 준비된 신경망으로 불러 넣어준다:
        // 이는 안면 탐지가 아닌 "사람 탐지"을 위해 사용된다.
        odNet = readNetFromCaffe(odConfigFile, odWeightFile);
        // CAFFE의 설정파일과 모델파일을 OpenCV에서 미리 준비된 신경망으로 불러 넣어준다:
        // 이 신경망은 "안면 탐지"를 위해 사용된다. dlib의 face detector보다 좋은 성능을 보인다.
        fdnet = readNetFromCaffe(fdConfigFile, fdWeightFile);
        odNet.setPreferableTarget(DNN_TARGET_OPENCL);
        fdnet.setPreferableTarget(DNN_TARGET_OPENCL);
        this->classesFile = classesFile;
        readClasses();
    }

    void     Recognizer::readClasses(){
        // 클래스 목록을 수집한다.
        /*
            >> `cv::String::c_str()`: "std::string::c_str()"로서도 함수가 존재;
                ...NULL(즉, 띄어쓰기 혹은 줄바꿈)이 있을 때마다 문자열을 나누어 행렬로 반환.
            >> `cv::vector::push_back(<data>)`: <data>를 현재 벡터의 맨 마지막으로 넣어준다.
                ...마치 스택 자료구조의 PUSH-BACK이라고 생각하면 된다.
                    결과적으로, 모든 클래스 종류는 "classes"라는 벡터 변수에 저장된다.
        */
        ifstream ifs(classesFile.c_str());
        string line;
        while(getline(ifs,line))
            classes.push_back(line);

    };

    bool     Recognizer::humanDetection(Mat& frame){
        bool found = true;
        // 모델의 물체인식을 위해 cv::Mat 형태의 프레임을 "BLOB" 형태로 변형시킨다.
        /*
            >> `blobFromImage(<input>, <output>, <scalefactor>, <size>, <mean>, <swapRB>, <crop>, <ddepth>)`
                ...(1/255): 픽셀값 0~255를 정규화된 RGB 값 0~1로 만들기 위해 값을 스케일한다.
                ...cv::Size(300,300): 모델의 .prototxt 구성(설정)파일에서 언급한 Blob 크기를 맞추기 위해 출력되는 blob 사이즈를 300x300으로 변경.
        */
        cv::Mat blob = blobFromImage(frame, 0.007843, cv::Size(inWidth,inHeight), 127.5);

        // HAVE BLOB AS AN INPUT OF THE NEURAL NETWORK TO PASS THROUGH (PLACED BUT NOT PASSED THROUGH YET).
        odNet.setInput(blob);

        // RUN FORWARD PASS TO COMPUTE OUTPUT OF LAYER WITH NAME "outNames": forward() [3/4]
        /*
            >> IF "outNames" is empty, the `cv::dnn::Net::forward()` runs forward pass for the whole network.
                ...returns variable "outs" which contains blobs for first outputs of specified layers.
            >> Variable "object_detection":
                ...rank-1: None
                ...rank-2: None
                ...rank-3: number of object detected.
                ...rank-4: element < ? >, <label>, <confidence>, <coord_x1>, <coord_y1>, <coord_x2>, <coord_y2>
        */
        cv::Mat detection = odNet.forward();

        for(int i =0; i < detection.size.p[2]; i++){
            float confidence = detection.at<float>(Vec<int,4>(0,0,i,2));

            if(confidence > 0.6) {
                int idx = detection.at<float>(Vec<int, 4>(0, 0, i, 1));
                String label = classes[idx];

                if (label.compare("person")) {
                    found =false;
                    continue;
                }
                int startX = (int) (detection.at<float>(Vec<int,4>(0,0,i,3)) * frame.cols);
                int startY = (int) (detection.at<float>(Vec<int,4>(0,0,i,4)) * frame.rows);
                int endX = (int) (detection.at<float>(Vec<int,4>(0,0,i,5)) * frame.cols);
                int endY = (int) (detection.at<float>(Vec<int,4>(0,0,i,6)) * frame.rows);

                cv::rectangle(frame, Point(startX,startY), Point(endX,endY),Scalar(0, 255, 0),2);
                putText(frame, label, Point(startX, startY-15), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0,255,0),2);

            }
        }
        return found;
    }


    std::vector<dlib::rectangle>     Recognizer::faceDetection(Mat& frame){
        std::vector<dlib::rectangle> locations;
        int frameHeight = frame.rows;
        int frameWidth = frame.cols;


        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);


        fdnet.setInput(inputBlob,"data");
        cv::Mat detection = fdnet.forward("detection_out");

        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

                locations.push_back(dlib::rectangle(x1,y1,x2,y2));
            }
        }
        return locations;
    }

    dlib::array<matrix<rgb_pixel>>     Recognizer::faceLandMark(Mat& frame,std::vector<dlib::rectangle>& locations) {
        // FACIAL LANDMARK DETECTION
        /*
            >> `dlib::shape_predictor`: an empty DNN frame that can take external model such as "shape_predictor_68_face_landmarks.dat".
                ...returns set of point locations that define the pose of the object, in this case, face landmark locations.
                ...therefore, shape_predictor is a neural network for "FACIAL DETECTION".
                    ...There exist `dlib::shape_predictor_trainer` which can train custom model from an empty DNN frame.
            >> `dlib::deserialize()`  : recover data saved in serialized back to its original format with deserialization.
                ...the operator `>>` imports the file of model to the "dlib::shape_predictor" object.
            >> `shape_predictor_68_face_landmarks.dat`: a file containing a model of pin-pointing 68 facial landmark.
                ...data is saved in serialized format which can be stored, transmitted and reconstructed (deserialization) later.
                ...얼굴 랜드마크 데이터 파일 "shape_predictor_68_face_landmarks.dat"의 라이선스는 함부로 상업적 이용을 금합니다.
                ...본 파일을 상업적 용도로 사용하기 위해서는 반드시 Imperial College London에 연락하기 바랍니다.
        */
        // STORES FACIAL IMAGE CHIP(AKA. FACIAL PORTION OF CROPPED IMAGE) FOR FACIAL RECOGNITION.
        /*
            >> `auto shape = net_landmark(<image>, <face_rect>)`: determines the facial region with <face_rect> in the  original image <image>,
                ...and from there extract 68-landmarks.
                ...object "shape" has rect data and 68-landmarks data (which includes x,y pixel locations).
            >> `dlib::get_face_chip_details(<landmark_data>,<crop_size(square)>,<crop_padding>)`: considering <crop_size> and <crop_padding>,
                ...provides chip_detail object to `dlib::extract_image_chip()` for chipping reference based on landmark 5-point or 68-point.
            >> `dlib::extract_image_chip(<input_image>,<chip_details>,<output_chip>)`: while <chip_details> works as a image-crop reference,
                ...extract image_chip from <input_image> to <output_chip>.
        */
        matrix<rgb_pixel> img;
        cv::Mat rgb_frame;
        dlib::assign_image(img, dlib::cv_image<rgb_pixel>(frame));
        dlib::array<matrix<rgb_pixel>> result;
        for (auto face : locations) {
            auto shape = pose_model(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            result.push_back(move(face_chip));
        }
        return result;
    }

    std::vector<matrix<float,0,1>>     Recognizer::faceEncoding(dlib::array<matrix<rgb_pixel>>& faces){

        return faceEncoder(faces,16);
    }

//    float calDistance(matrix<float, 0, 1> descriptors1, matrix<float, 0, 1> descriptors2){
//        return length(descriptors1-descriptors2);
//    }

