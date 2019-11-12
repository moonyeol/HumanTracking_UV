//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//Í∏∞Ï°¥ face_recognition_dlib
//?úÑ 3Í∞? Ï∞∏Í≥†
//
//Ïª¥Ìåå?ùº Î∞?
//g++ -std=c++11 -O3 -I.. /home/eon/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o main main.cpp $(pkg-config opencv4 --libs --cflags)
#include <opencv2/videoio.hpp>


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
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include <thread>

/*__________ RPLIDAR __________*/
#include <rplidar.h>
#include "rplidar.hpp"
#include <cmath>


using namespace cv;
using namespace cv::dnn;
using namespace dlib;
using namespace std;
using namespace rp::standalone::rplidar;


#define CYCLE 360       // ?ïú ?Ç¨?ù¥?Å¥??? 360?èÑ.
#define DIRECTION 4     // ?ù¥?èôÎ∞©Ìñ• Í∞úÏàò.

#define LEFT "l"
#define RIGHT "r"
#define GO "g"
#define BACK "b"
#define STOP "s"


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;


template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;


template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;


template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;


template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;


using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);


const std::string caffeConfigFile = "models/deploy.prototxt";
const std::string caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";


const std::string  encodeDat = "models/dlib_face_recognition_resnet_model_v1.dat";
const std::string  landmarkDat = "models/shape_predictor_68_face_landmarks.dat";
const std::string  odConfigFile = "models/MobileNetSSD_deploy.prototxt";
const std::string  odWeightFile = "models/MobileNetSSD_deploy.caffemodel";
const std::string  fdConfigFile = "models/deploy.prototxt";
const std::string  fdWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string  classesFile = "names";
const std::string  userImg = "pic/user.jpg";
// CAFFE?ùò PROTOTXT??? CAFFEMODEL?äî DARKNET?ùò CONFIG??? WEIGHT ?åå?ùºÍ≥? ?èô?ùº?ïò?ã§ Ï¢ÖÎ•ò?ùò ?åå?ùº?ù¥?ã§;
long tempsize=0;
char* data = STOP;
bool isEnd = false;

class Recognizer{

private:
    shape_predictor pose_model;
    anet_type faceEncoder;
    Net odNet;
    Net fdnet;
    String classesFile;
    std::vector<string> classes;
public:

    Recognizer(String landmarkDat, String frNetModel, string odConfigFile, string odWeightFile, String fdConfigFile, String fdWeightFile, String classesFile){
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
        // CAFFE?ùò ?Ñ§?†ï?åå?ùºÍ≥? Î™®Îç∏?åå?ùº?ùÑ OpenCV?óê?Ñú ÎØ∏Î¶¨ Ï§?ÎπÑÎêú ?ã†Í≤ΩÎßù?úºÎ°? Î∂àÎü¨ ?Ñ£?ñ¥Ï§??ã§:
        // ?ù¥?äî ?ïàÎ©? ?ÉêÏß?Í∞? ?ïÑ?ãå "?Ç¨?ûå ?ÉêÏß?"?ùÑ ?úÑ?ï¥ ?Ç¨?ö©?êú?ã§.
        odNet = readNetFromCaffe(odConfigFile, odWeightFile);
        // CAFFE?ùò ?Ñ§?†ï?åå?ùºÍ≥? Î™®Îç∏?åå?ùº?ùÑ OpenCV?óê?Ñú ÎØ∏Î¶¨ Ï§?ÎπÑÎêú ?ã†Í≤ΩÎßù?úºÎ°? Î∂àÎü¨ ?Ñ£?ñ¥Ï§??ã§:
        // ?ù¥ ?ã†Í≤ΩÎßù??? "?ïàÎ©? ?ÉêÏß?"Î•? ?úÑ?ï¥ ?Ç¨?ö©?êú?ã§. dlib?ùò face detectorÎ≥¥Îã§ Ï¢ãÏ?? ?Ñ±?ä•?ùÑ Î≥¥Ïù∏?ã§.
        fdnet = readNetFromCaffe(fdConfigFile, fdWeightFile);
        odNet.setPreferableTarget(DNN_TARGET_OPENCL);
        fdnet.setPreferableTarget(DNN_TARGET_OPENCL);
        this->classesFile = classesFile;
        readClasses();
    }

    void readClasses(){
        // ?Å¥?ûò?ä§ Î™©Î°ù?ùÑ ?àòÏßëÌïú?ã§.
        /*
            >> `cv::String::c_str()`: "std::string::c_str()"Î°úÏÑú?èÑ ?ï®?àòÍ∞? Ï°¥Ïû¨;
                ...NULL(Ï¶?, ?ùÑ?ñ¥?ì∞Í∏? ?òπ??? Ï§ÑÎ∞îÍø?)?ù¥ ?ûà?ùÑ ?ïåÎßàÎã§ Î¨∏Ïûê?ó¥?ùÑ ?Çò?àÑ?ñ¥ ?ñâ?†¨Î°? Î∞òÌôò.
            >> `cv::vector::push_back(<data>)`: <data>Î•? ?òÑ?û¨ Î≤°ÌÑ∞?ùò Îß? ÎßàÏ??ÎßâÏúºÎ°? ?Ñ£?ñ¥Ï§??ã§.
                ...ÎßàÏπò ?ä§?Éù ?ûêÎ£åÍµ¨Ï°∞Ïùò PUSH-BACK?ù¥?ùºÍ≥? ?ÉùÍ∞ÅÌïòÎ©? ?êú?ã§.
                    Í≤∞Í≥º?†Å?úºÎ°?, Î™®Îì† ?Å¥?ûò?ä§ Ï¢ÖÎ•ò?äî "classes"?ùº?äî Î≤°ÌÑ∞ Î≥??àò?óê ????û•?êú?ã§.
        */
        ifstream ifs(classesFile.c_str());
        string line;
        while(getline(ifs,line))
            classes.push_back(line);

    };

    bool humanDetection(Mat& frame){
        bool found = true;
        // Î™®Îç∏?ùò Î¨ºÏ≤¥?ù∏?ãù?ùÑ ?úÑ?ï¥ cv::Mat ?òï?Éú?ùò ?îÑ?†à?ûÑ?ùÑ "BLOB" ?òï?ÉúÎ°? Î≥??òï?ãú?Ç®?ã§.
        /*
            >> `blobFromImage(<input>, <output>, <scalefactor>, <size>, <mean>, <swapRB>, <crop>, <ddepth>)`
                ...(1/255): ?îΩ???Í∞? 0~255Î•? ?†ïÍ∑úÌôî?êú RGB Í∞? 0~1Î°? ÎßåÎì§Í∏? ?úÑ?ï¥ Í∞íÏùÑ ?ä§Ïº??ùº?ïú?ã§.
                ...cv::Size(300,300): Î™®Îç∏?ùò .prototxt Íµ¨ÏÑ±(?Ñ§?†ï)?åå?ùº?óê?Ñú ?ñ∏Í∏âÌïú Blob ?Å¨Í∏∞Î?? ÎßûÏ∂îÍ∏? ?úÑ?ï¥ Ï∂úÎ†•?êò?äî blob ?Ç¨?ù¥Ï¶àÎ?? 300x300?úºÎ°? Î≥?Í≤?.
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


    std::vector<dlib::rectangle> faceDetection(Mat& frame){
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

    dlib::array<matrix<rgb_pixel>> faceLandMark(Mat& frame,std::vector<dlib::rectangle>& locations) {
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
                ...?ñºÍµ? ?ûú?ìúÎßàÌÅ¨ ?ç∞?ù¥?Ñ∞ ?åå?ùº "shape_predictor_68_face_landmarks.dat"?ùò ?ùº?ù¥?Ñ†?ä§?äî ?ï®Î∂?Î°? ?ÉÅ?óÖ?†Å ?ù¥?ö©?ùÑ Í∏àÌï©?ãà?ã§.
                ...Î≥? ?åå?ùº?ùÑ ?ÉÅ?óÖ?†Å ?ö©?èÑÎ°? ?Ç¨?ö©?ïòÍ∏? ?úÑ?ï¥?Ñú?äî Î∞òÎìú?ãú Imperial College London?óê ?ó∞?ùΩ?ïòÍ∏? Î∞îÎûç?ãà?ã§.
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

    std::vector<matrix<float,0,1>> faceEncoding(dlib::array<matrix<rgb_pixel>>& faces){

        return faceEncoder(faces,16);
    }

//    float calDistance(matrix<float, 0, 1> descriptors1, matrix<float, 0, 1> descriptors2){
//        return length(descriptors1-descriptors2);
//    }

};


void humanTracking(){
    try {   // TRY BLOCK CODE START: WHOLE PROCESS FOR DETECTION AND AUTOMATION.
        Recognizer recognizer(landmarkDat,encodeDat,odConfigFile,odWeightFile,fdConfigFile,fdWeightFile,classesFile);

        // CREATE VECTOR OBJECT CALLED "detection1" WHICH CAN CONTAIN LIST OF MAT OBJECTS.
        /*
            >> `std::vector< std::vector< std::vector<Mat> > >`: Creates 3D vector array containing Mat data-type.
        */
        Mat frame;
        cv::VideoCapture cap;

        // OPEN DEFAULT CAMERA OF `/dev/video0` WHERE ITS INTEGER IS FROM THE BACK.
        /*
            Set the video resolution by `cap` as designated pixel size, does not work on called video file.
            Access, or "open" default camera which is presumes to be `/dev/video0` file
                ...(which is where number 0 may have derived from).
        */

        cap.open(1); // ?Ö∏?ä∏Î∂? Ïπ¥Î©î?ùº?äî cap.open(1) ?òê?äî cap.open(-1)
        // USB Ïπ¥Î©î?ùº?äî cap.open(0);
        int x1;
        int y1;
        int x2;
        int y2;
        bool found = true;// FIXME INEFFICIENT CODE
        int countFrame = 0;
        // Load face detection and pose estimation models.

        // __________ PREPARATION OF FACIAL RECOGNITION BY SETTING NEURAL NETWORK FOR FACIAL DETECTION AND RECOGNITION. __________ //









        // __________ PROCESS OF ACQUIRING USER FACIAL DATA FOR A FUTURE USER RECOGNITION. __________ //


        cv::Mat user_img = cv::imread(userImg,cv::IMREAD_COLOR);
        std::vector<dlib::rectangle> locations = recognizer.faceDetection(user_img);
        // PREPARE A VARIABLE TO STORE ALL OF DETECTED FACES.
        /*
            >> `dlib::array<dlib::matrix<dlib::rgb_pixel>>`: dlib::matrix<dlib::rgb_pixel> has been discussed above.
                ...Since dlib::array store rank-1 tensor, the parameterized type is to store multiple number of image data in a list.
        */
        dlib::array<matrix<rgb_pixel>> faces = recognizer.faceLandMark(user_img,locations);

        // CREATE A VARIALBE "face_detected_user" FOR FUTURE FACIAL COMPARISON.
        /*
            >> It is still unclear what the stored value `face_detected_user[0]` represents,
                ...but it is possible the value is (1) Loss, (2) Score, or (3) Confidence.
            >> `net_recognition(<input_data>,<batch_size>)`: uncertain of a purpose of a <batch_size> is; there's only one facial data here!
        */
        std::vector<matrix<float,0,1>> face_descriptors = recognizer.faceEncoding(faces);



        // __________ PROCESS OF PERSON DETECTION USING EXISTING FRAMEWORK MODEL. __________ //





        // ?õπÏ∫†ÏúºÎ°? Ï¥¨ÏòÅ?ù¥ ÏßÑÌñâ?êò?äî ?èô?ïà...
        while(cap.isOpened()){




            // VIDEOCAPTURE ?Å¥?ûò?ä§?ùò "CAPTURE"?äî Ï¥¨ÏòÅ?êú ?àúÍ∞ÑÏùò ?îÑ?†à?ûÑ?ùÑ cv::Mat ?òï?Éú?ùò "FRAME" ?ò§Î∏åÏ†ù?ä∏?óê ?ï†?ãπ?ïú?ã§.
            cap >> frame;
            double t = cv::getTickCount();
            resize(frame,frame, Size(640,480));

            //found = recognizer.humanDetection(frame);
            found = false;



            
            if(countFrame%3==0) {   // START OF OUTER IF CONDITION
                //face recoginition Íµ¨ÌòÑ Ï§?
                //if (found) {    // START OF INNER IF CONDITION.



                    std::vector<dlib::rectangle> locations2 = recognizer.faceDetection(frame);


                    dlib::array<matrix<rgb_pixel>> faces2 = recognizer.faceLandMark(frame,locations2);

                    std::vector<matrix<float, 0, 1>> face_descriptors2 = recognizer.faceEncoding(faces2);
                    std::vector<String> names;
                    String name;

                    // START OF FOR LOOP: USER DETECTION AND LOCATION FINDER.
                    for (size_t i = 0; i < face_descriptors2.size(); ++i) {
                        name = "unknown";
                        if (length(face_descriptors[0] - face_descriptors2[i])< 0.5) {
                            found = true;
                            name = "user";
                            long xcenter = (locations2[i].right() + locations2[i].left()) / 2;
                            long ycenter = (locations2[i].bottom() + locations2[i].top()) / 2;
                            long size = (locations2[i].right() - locations2[i].left());


                           
                            if (xcenter <180)
                            {
                                data = LEFT;
                            }
                            else if (xcenter > 400)
                            {
                                data = RIGHT;
                            }
                            else if (size>71)
                            {

                                data = BACK;
                            }
                            else if (size<69)
                            {
                                data = GO;
                            }
                            else
                            {
                                data = STOP;
                            }
                            cout << "size = " << size << endl;
                            cout <<"data(main) = "<< *data<<endl;
                        }   // END OF FOR LOOP: USER DETECTION AND LOCATION FINDER.
                        if(!found){data = STOP;}

                        //cout<<"data = "<<data<<endl;
                        names.push_back(name);

                    }   // END OF INNER IF CONDITION.

                    int i = 0;


                    for (int i = 0; i < locations2.size(); i++) {
                        cv::rectangle(frame, Point(locations2[i].left(), locations2[i].top()), Point(locations2[i].right(), locations2[i].bottom()), Scalar(0, 255, 0), 2);
                        putText(frame, names[i], Point(locations2[i].left() + 6, locations2[i].top() - 6), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 2);

                    }

               // }   // END OF OUTER IF CONDITION


            }   // END OF WHILE LOOP
            double tt_opencvDNN = 0;
            double fpsOpencvDNN = 0;



            tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fpsOpencvDNN = 1/tt_opencvDNN;
            putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

            // ?õπÏ∫†Ïóê?Ñú Ï¥¨ÏòÅ?ïò?äî ?òÅ?ÉÅ?ùÑ Î≥¥Ïó¨Ï§??ã§; Enter ?Ç§Î•? ?àÑÎ•¥Î©¥ Ï¢ÖÎ£å.
            //cv::imshow("HumanTrackingUV",frame);
            if (cv::waitKey(30)==13) break;
		char key = getch();
		if(key == 'q') break;
            countFrame++;


        }// END OF WHILE LOOP
    }// END OF TRY BLOCK: WHOLE PROCESS FOR DETECTION AND AUTOMATION.

        // ?òà?ô∏Ï≤òÎ¶¨ 1: ?ûú?ìúÎßàÌÅ¨ ÎßàÌÅ¨ Î™®Îç∏?ùÑ Ï∞æÏùÑ ?àò ?óÜ?äµ?ãà?ã§.
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }

        // ?òà?ô∏Ï≤òÎ¶¨ 2
    catch(exception& e)
    {
        cout << e.what() << endl;
    }

    isEnd = true;
}
void lidar(int fd) {
    class rplidar rplidarA1;
    char *move;
    while (!isEnd) {
        rplidarA1.scan();
        rplidarA1.retrieve();
        move = rplidarA1.returnMove(data);
        rplidarA1.result();
        write(fd, data, strlen(data));
    }
}
int main(int argc, char **argv ) {



    int fd;
    fd=open("/dev/ttyACM0", O_RDWR | O_NOCTTY );  // Ïª®Ìä∏Î°? c Î°? Ï∑®ÏÜå?ïà?êòÍ≤? ?ïòÍ∏? | O_NOCTTY

    //struct termios newtio;
    struct termios toptions;




    //fprintf(stderr,"init_serialport: opening port %s @ %d bps\n",

    //        serialport,baud);



    //fd = open(serialport, O_RDWR | O_NOCTTY | O_NDELAY);

    if (fd == -1)  {

        perror("init_serialport: Unable to open port ");

        return -1;

    }



    if (tcgetattr(fd, &toptions) < 0) {

        perror("init_serialport: Couldn't get term attributes");

        return -1;

    }




    // 8N1

    toptions.c_cflag &= ~PARENB;//Enable parity generation on output and parity checking for input.

    toptions.c_cflag &= ~CSTOPB;//Set two stop bits, rather than one.

    toptions.c_cflag &= ~CSIZE;//Character size mask.  Values are CS5, CS6, CS7, or CS8.



    // no flow control

    toptions.c_cflag &= ~CRTSCTS;//(not in POSIX) Enable RTS/CTS (hardware) flow control. [requires _BSD_SOURCE or _SVID_SOURCE]

    toptions.c_cflag = B115200 | CS8 | CLOCAL | CREAD; // CLOCAL : Ignore modem control lines CREAD :Enable receiver.

    toptions.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl
    toptions.c_iflag = IGNPAR | ICRNL;

    toptions.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Enable canonical mode (described below)./Echo input characters.
    // If ICANON is also set, the ERASE character erases the preced??êing input character, and WERASE erases the preceding word.
    // When any of the characters INTR, QUIT, SUSP, or DSUSP are received, generate the corresponding signal.

    toptions.c_oflag &= ~OPOST; //Enable implementation-defined output processing.

    // see: http://unixwiz.net/techtips/termios-vmin-vtime.html
    toptions.c_cc[VMIN]  = 0;
    toptions.c_cc[VTIME] = 20;

    if( tcsetattr(fd, TCSANOW, &toptions) < 0) {
        perror("init_serialport: Couldn't set term attributes");
        return -1;
    }




    thread hThread(humanTracking);
    thread lThread{lidar,fd};
    hThread.join();
    lThread.join();







    // ?òÅ?ÉÅ?ù∏?ãùÍ≥? ?ûê?ú®Ï£ºÌñâ?ù¥ Î™®Îëê ?Åù?ÇòÎ©? R/W ?åå?ùº?ùÑ ?ã´?äî?ã§.
    close(fd);

    // ?òÅ?ÉÅ?ù∏?ãùÍ≥? ?ûê?ú®Ï£ºÌñâ?ù¥ Î™®Îëê ?Åù?Ç¨?úºÎ©? OpenCV Ï∞ΩÏùÑ ?ã´?äî?ã§.
    cv::destroyWindow("HumanTrackingUV");

    return 0;
}
