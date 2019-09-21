//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고
//
//컴파일 방
//g++ -std=c++11 -O3 -I.. /home/eon/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o main main.cpp $(pkg-config opencv4 --libs --cflags)


#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/objdetect.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/dnn/dict.hpp>
#include <opencv4/opencv2/dnn/layer.hpp>
#include <opencv4/opencv2/dnn/dnn.inl.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/core/cuda.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <dlib/geometry/vector.h>
#include <dlib/matrix.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>    /* Standard input/output definitions */
#include <stdlib.h>
#include <sstream>
#include <string>
#include <stdint.h>   /* Standard types */
#include <string.h>   /* String function definitions */
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <errno.h>    /* Error number definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include <sys/ioctl.h>
#include <getopt.h>

using namespace cv;
using namespace cv::dnn;
using namespace dlib;
using namespace std;


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



//template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
//template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
//
//template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
//template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;
//
//using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

#define CAFFE

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
long tempsize=0;

//const std::string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
//const std::string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";









int main()
{
    // TRY BLOCK CODE START

    int fd;
    fd=open("/dev/ttyACM1", O_RDWR | O_NOCTTY );  // 컨트롤 c 로 취소안되게 하기 | O_NOCTTY

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

    //speed_t brate = baud; // let you override switch below if needed



    //cfsetispeed(&toptions, brate);

    //cfsetospeed(&toptions, brate);



    // 8N1

    toptions.c_cflag &= ~PARENB;//Enable parity generation on output and parity checking for input.

    toptions.c_cflag &= ~CSTOPB;//Set two stop bits, rather than one.

    toptions.c_cflag &= ~CSIZE;//Character size mask.  Values are CS5, CS6, CS7, or CS8.



    // no flow control

    toptions.c_cflag &= ~CRTSCTS;//(not in POSIX) Enable RTS/CTS (hardware) flow control. [requires _BSD_SOURCE or _SVID_SOURCE]



    toptions.c_cflag = B9600 | CS8 | CLOCAL | CREAD; // CLOCAL : Ignore modem control lines CREAD :Enable receiver.
    //toptions.c_cflag |= CREAD | CLOCAL;  // turn on READ & ignore ctrl lines

    toptions.c_iflag &= ~(IXON | IXOFF | IXANY); // turn off s/w flow ctrl
    toptions.c_iflag = IGNPAR | ICRNL;



    toptions.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Enable canonical mode (described below)./Echo input characters.
    // If ICANON is also set, the ERASE character erases the preced‐ing input character, and WERASE erases the preceding word.
    // When any of the characters INTR, QUIT, SUSP, or DSUSP are received, generate the corresponding signal.

    toptions.c_oflag &= ~OPOST; //Enable implementation-defined output processing.



    // see: http://unixwiz.net/techtips/termios-vmin-vtime.html

    toptions.c_cc[VMIN]  = 0;

    toptions.c_cc[VTIME] = 20;



    if( tcsetattr(fd, TCSANOW, &toptions) < 0) {

        perror("init_serialport: Couldn't set term attributes");

        return -1;

    }
    try
    {

        // CREATE VECTOR OBJECT CALLED "detection1" WHICH CAN CONTAIN LIST OF MAT OBJECTS.
        /*
            >> `std::vector< std::vector< std::vector<Mat> > >`: Creates 3D vector array containing Mat data-type.
        */
        Mat frame, blob, grayFrame;
        std::vector<Mat> detection1;
        std::vector<std::vector<std::vector<Mat>>> detection2;
        cv::VideoCapture cap;

        // OPEN DEFAULT CAMERA OF `/dev/video0` WHERE ITS INTEGER IS FROM THE BACK.
        /*
            Set the video resolution by `cap` as designated pixel size, does not work on called video file.
            Access, or "open" default camera which is presumes to be `/dev/video0` file
                ...(which is where number 0 may have derived from).
        */

        cap.open(0); // 노트북 카메라는 cap.open(1) 또는 cap.open(-1)
        // USB 카메라는 cap.open(0);
        int x1;
        int y1;
        int x2;
        int y2;
        bool found = false;// FIXME INEFFICIENT CODE

        // Load face detection and pose estimation models.

        // __________ PREPARATION OF FACIAL RECOGNITION BY SETTING NEURAL NETWORK FOR FACIAL DETECTION AND RECOGNITION. __________ //

        // DECLARE A FRONTAL FACE DETECTOR (HOG BASED) TO A VARIABLE "detector".
        /*
            >> `dlib::get_frontal_face_detector()`: return "dlib::object_detector" object upon detecting a frontal face.
                ...while the returned variable type here is `dlib::frontal_face_detector` is actually an alias of `dlib::object_detector`.
                ...Python didn't needed this data-type since variable in Python does not need to data-type designation.

            >> `dlib::object_detector`: a tool for detecting the positions of objects in an image.
                ...returns `dlib::rectangle` describing left, top, right, and bottom boundary pixel position of the rectangle.
        */





        frontal_face_detector detector = get_frontal_face_detector();
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
        shape_predictor pose_model;
        deserialize("models/shape_predictor_68_face_landmarks.dat") >> pose_model;
        // ASSIGN VARIABLE "net" AS AN OBJECT OF "anet_type" DEFINED ABOVE.
        /*
            >> `models/dlib_face_recognition_resnet_model_v1.dat`: DNN for a "FACIAL RECOGNITION".
                ...it is presume this file too needs deserialization which reconstructs to original data format.
                ...for more information of a serialization, read this webpage; https://www.geeksforgeeks.org/serialization-in-java/.

            >> statement `dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;`
                ...reconstructed recognition model is placed in a hollow neural network frame manually created using operator `>>`.
                ...where now this variable `net` works as a model.
        */

//        face_recognition_model_v1 face_encoder =  face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat");
        anet_type net;
        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;
//        deserialize("models/mmod_human_face_detector.dat") >> net;


//        net_type detector;
//        deserialize("mmod_human_face_detector.dat") >> detector;


#ifdef CAFFE
        Net net2 = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
        Net net2 = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif
        net2.setPreferableTarget(DNN_TARGET_OPENCL);

        // __________ PROCESS OF ACQUIRING USER FACIAL DATA FOR A FUTURE USER RECOGNITION. __________ //

        // ACQUIRE A FACIAL IMAGE FOR RECOGNITION.
        /*
            >> `dlib::matrix<dlib::rgb_pixel>`: dlib::matrix is a rank-2 tensor (width and height)
                ... and dlib::rgb_pixel is a rank-1 tensor (RGB channel per pixel). Therefore, the parameterized type is to stores image data.
        */

        // PREPARE A VARIABLE TO STORE ALL OF DETECTED FACES.
        /*
            >> `dlib::array<dlib::matrix<dlib::rgb_pixel>>`: dlib::matrix<dlib::rgb_pixel> has been discussed above.
                ...Since dlib::array store rank-1 tensor, the parameterized type is to store multiple number of image data in a list.
        */
        dlib::array<matrix<rgb_pixel>> faces;

        // STORES FACIAL IMAGE CHIP(AKA. FACIAL PORTION OF CROPPED IMAGE) FOR FACIAL RECOGNITION.
        /*
            >> `for (auto face : facial_detector(<image>))`: a range-based for loop, iterating as many as number of detected face in FACIAL IMAGE.
                ...returns dlib::rectangle of frontal face starting from left, top, right, and bottom boundary pixel position.

            >> `auto shape = net_landmark(<image>, <face_rect>)`: determines the facial region with <face_rect> in the  original image <image>,
                ...and from there extract 68-landmarks.
                ...object "shape" has rect data and 68-landmarks data (which includes x,y pixel locations).

            >> `dlib::get_face_chip_details(<landmark_data>,<crop_size(square)>,<crop_padding>)`: considering <crop_size> and <crop_padding>,
                ...provides chip_detail object to `dlib::extract_image_chip()` for chipping reference based on landmark 5-point or 68-point.

            >> `dlib::extract_image_chip(<input_image>,<chip_details>,<output_chip>)`: while <chip_details> works as a image-crop reference,
                ...extract image_chip from <input_image> to <output_chip>.
        */

        //face recoginition 구현 중

        matrix<rgb_pixel> user_img;
        load_image(user_img, "user.jpg");

        for (auto face : detector(user_img)) {
            auto shape = pose_model(user_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(user_img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }


        // CREATE A VARIALBE "face_detected_user" FOR FUTURE FACIAL COMPARISON.
        /*
            >> It is still unclear what the stored value `face_detected_user[0]` represents,
                ...but it is possible the value is (1) Loss, (2) Score, or (3) Confidence.

            >> `net_recognition(<input_data>,<batch_size>)`: uncertain of a purpose of a <batch_size> is; there's only one facial data here!
        */
        std::vector<matrix<float,0,1>> face_descriptors = net(faces,16);



        // __________ PROCESS OF PERSON DETECTION USING EXISTING FRAMEWORK MODEL. __________ //

        // PROTOTXT AND CAFFEMODEL IS A COUNTERPART OF CONFIG AND WEIGHT IN DARKNET;
        // ...FOR PERSON DETECTION (NOT A FACIAL DETECTION)
        String prototxt = "models/MobileNetSSD_deploy.prototxt";
        String model = "models/MobileNetSSD_deploy.caffemodel";



        // ACQUIRE LIST OF CLASSES.
        /*
            >> `cv::String::c_str()`: also available as "std::string::c_str()";
                ...returns array with strings splitted on NULL (blank space, new line).

            >> `cv::vector::push_back(<data>)`: push the data at the back end of the current last element.
                ...just like a push-back of the stack data structure.

                    Hence, the name of the classes are all stored in variable "classes".
        */
        String classesFile = "names";
        ifstream ifs(classesFile.c_str());
        string line;
        std::vector<string> classes;
        while(getline(ifs,line))
            classes.push_back(line);

        // IMPORT CAFFE CONFIG AND MODEL TO THE NEURAL NETWORK:
        //  ...for a "PERSON DETECTION".

        Net net1 = readNetFromCaffe(prototxt, model);
        net1.setPreferableTarget(DNN_TARGET_OPENCL);

        // WHILE CAMERA IS OPENED...
        while(cap.isOpened()){
            // VIDEOCAPTURE "CAPTURE" RETURN ITS FRAME TO MAT "FRAME".

            cap >> frame;
            double t = cv::getTickCount();
            resize(frame,frame, Size(640,480));

            // CONVERT FRAME TO PREPROCESSED "BLOB" FOR MODEL PREDICTION.
            /*
                >> `blobFromImage(<input>, <output>, <scalefactor>, <size>, <mean>, <swapRB>, <crop>, <ddepth>)`
                    ...(1/255): Scale the factors as to normalize RGB value from (0~255) to (0~1).
                    ...cv::Size(300,300): resize the output blob to 300x300 as noted by model configuration .prototxt file.
            */
            blob = blobFromImage(frame, 0.007843, Size(300,300), 127.5);

            // HAVE BLOB AS AN INPUT OF THE NEURAL NETWORK TO PASS THROUGH (PLACED BUT NOT PASSED THROUGH YET).
            net1.setInput(blob);

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
            Mat detection = net1.forward();


            found =true;
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect> boxes;


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






            //face recoginition 구현 중
            if(found) {
                char *data;

                matrix<rgb_pixel> img;
//                cv::cuda::GpuMat rgb_frame;
//                Mat rgb_frame;
//                cvtColor(frame,rgb_frame,COLOR_BGR2RGB);

                dlib::assign_image(img, dlib::cv_image<rgb_pixel>(frame));




                std::vector<dlib::rectangle> locations;

                int frameHeight = frame.rows;
                int frameWidth = frame.cols;

#ifdef CAFFE
                cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
                cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

                net2.setInput(inputBlob,"data");
                cv::Mat detection = net2.forward("detection_out");

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




                dlib::array<matrix<rgb_pixel>> faces2;
//                auto locations = detector(img);
                for (auto face : locations) {
                    auto shape = pose_model(img, face);
                    matrix<rgb_pixel> face_chip;
                    extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                    faces2.push_back(move(face_chip));
                }
                std::vector<matrix<float, 0, 1>> face_descriptors2 = net(faces2,16);
                std::vector<String> names;
                String name;
                for (size_t i = 0; i < face_descriptors2.size(); ++i) {
                    name = "unknown";
                    if (length(face_descriptors[0] - face_descriptors2[i]) < 0.5) {
                        name = "user";
                        long xcenter = (locations[i].right() + locations[i].left())/2;
                        long ycenter= (locations[i].bottom() + locations[i].top())/2;

                        if (xcenter!=320&&ycenter!= 240)
                        {
                            cout<<"중심을 벗어낫습니다"<<endl;
                        }

                        if (tempsize==0){
                            tempsize = xcenter;
                            std::cout<<"값을 저장하였습니다"<<endl;
                        }
                        else
                        {
                            if (tempsize < xcenter)
                            {
                                cout<<"저장값 : "<<tempsize<<endl;
                                cout<<"대상이 가까워졌습니다."<<endl;
                                tempsize = xcenter;
                                data = "g";

                                //data= "g";
                                //write(fd, data, strlen(data));
                                write(fd, data, strlen(data));
                            }
                            else if(tempsize > xcenter)
                            {
                                cout<<"저장값 : "<<tempsize<<endl;
                                cout<<"대상이 멀어졌습니다."<<endl;
                                tempsize = xcenter;
                                data = "b";

                                //data = "b";
                                //write(fd, data, strlen(data));
                                write(fd, data, strlen(data));
                            }
                        }
                    }

                    cout<<"data = "<<data<<endl;
                    names.push_back(name);
                }
                int i = 0;

//                for (auto&& l : locations) {
//                    cv::rectangle(frame, Point(l.rect.left(), l.rect.top()),
//                                  Point(l.rect.right(), l.rect.bottom()), Scalar(0, 255, 0), 2);
//                    putText(frame, names[i], Point(l.rect.left() + 6, l.rect.top() - 6),
//                            FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 2);
//                    i++;
//                }



                for (int i = 0; i < locations.size(); i++) {
                    cv::rectangle(frame, Point(locations[i].left(), locations[i].top()),
                                  Point(locations[i].right(), locations[i].bottom()), Scalar(0, 255, 0), 2);
                    putText(frame, names[i], Point(locations[i].left() + 6, locations[i].top() - 6),
                            FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 2);
                }



            }



            double tt_opencvDNN = 0;
            double fpsOpencvDNN = 0;



            tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fpsOpencvDNN = 1/tt_opencvDNN;
            putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);


            // SHOW CAPTURED VIDEO.

            cv::imshow("HumanTrackingUV",frame);
            if (cv::waitKey(30)==27)
            {
                break;
            }
        }// END OF WHILE LOOP
    }// END OF TRY BLOCK

        // EXCEPTION 1: NO LANDMARKING MODEL DETECTED.
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
        // EXCEPTION 2
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
    // DESTROY WINDOWS UPON END OF EXECUTION.
    close(fd);
    cv::destroyWindow("HumanTrackingUV");
    return 0;
}

