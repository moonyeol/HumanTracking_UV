//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고
//
//컴파일 방
//g++ -std=c++11 -O3 -I.. /home/eon/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o main main.cpp $(pkg-config opencv4 --libs --cflags)

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

/*__________ RPLIDAR __________*/
#include <rplidar.h>
#include <cmath>


using namespace cv;
using namespace cv::dnn;
using namespace dlib;
using namespace std;
using namespace rp::standalone::rplidar;


/*__________ RPLIDAR 행동교정 함수선언 __________*/
char* rplidarBehavior(/*char, */char*, int*);


#define CYCLE 360       // 한 사이클은 360도.
#define DIRECTION 4     // 이동방향 개수.

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
long tempsize=0;

const std::string  encodeDat = "models/dlib_face_recognition_resnet_model_v1.dat";
const std::string  landmarkDat = "models/shape_predictor_68_face_landmarks.dat";
const std::string  odConfigFile = "models/MobileNetSSD_deploy.prototxt";
const std::string  odWeightFile = "models/MobileNetSSD_deploy.caffemodel";
const std::string  fdConfigFile = "models/deploy.prototxt";
const std::string  fdWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
const std::string  classesFile = "names";
const std::string  userImg = "pic/user.jpg";




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
        deserialize(frNetModel) >> faceEncoder;
        deserialize(landmarkDat) >> pose_model;
        odNet = readNetFromCaffe(odConfigFile, odWeightFile);
        fdnet = readNetFromCaffe(fdConfigFile, fdWeightFile);
        odNet.setPreferableTarget(DNN_TARGET_OPENCL);
        fdnet.setPreferableTarget(DNN_TARGET_OPENCL);
        this->classesFile = classesFile;
        readClasses();
    }

    void readClasses(){
        ifstream ifs(classesFile.c_str());
        string line;
        while(getline(ifs,line))
            classes.push_back(line);

    };

    bool humanDetection(Mat& frame){
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




int main(int argc, char **argv ) {

    

    int fd;
    fd=open("/dev/ttyACM0", O_RDWR | O_NOCTTY );  // 컨트롤 c 로 취소안되게 하기 | O_NOCTTY

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

    // RPLIDAR A1과 통신을 위한 장치 드라이버 생성. 제어는 드라이버를 통해서 진행된다: 예. rplidarA1 -> functionName().
    RPlidarDriver * rplidarA1 = RPlidarDriver::CreateDriver(DRIVER_TYPE_SERIALPORT);
        
    try {   // TRY BLOCK CODE START: WHOLE PROCESS FOR DETECTION AND AUTOMATION.

        Recognizer recognizer(landmarkDat,encodeDat,odConfigFile,odWeightFile,fdConfigFile,fdWeightFile,classesFile);



        /*__________ [START]: RPLIDAR A1 센서 제어 관련 설정: GKO95 작성 __________*/

        // 배열 <distances>는 각 방향마다 가지는 최소 스캔 결과값을 포함한다.
        int distances[DIRECTION]={0};
            
        // 시리얼 포트 경로 "/dev/ttyUSB0"를 통해
        /*
            >> `rp::standalone::rplidar::connet()`: RPLidar 드라이버를 연결할 RPLIDAR A1 장치와 어떤 시리얼 포트를 사용할 것인지,
                그리고 통신채널에서 송수률(baud rate)인 초당 최대 비트, 즉 bit/sec을 선택한다. 일반적으로 RPLIDAR 모델의baud rate는 115200으로 설정한다.
                ...만일 드라이버와 장치의 연결이 성공되었으면 숫자 0을 반환한다.
        */
        u_result result = rplidarA1->connect("/dev/ttyUSB0", 115200);
        
        // 연결이 성공하였으면 아래의 코드를 실행한다
        if (IS_OK(result)) {
            
            // RPLIDAR 모터 동작.
            rplidarA1 -> startMotor();
            
            // RPLIDAR에는 여러 종류의 스캔 모드가 있는데, 이 중에서 일반 스캔 모드를 실행한다.
            /*
            >> `rp::standalone::rplidar::startScanExpress(<force>,<use_TypicalScan>,<options>,<outUsedScanMode>)`:
                ...<force>           - 모터 작동 여부를 떠나 가ㅇ제(force)로 스캔 결과를 반환하도록 한다.
                ...<use_TypicalScan> - true는 일반 스캔모드(초당 8k 샘플링), false는 호환용 스캔모드(초당 2k 샘플링).
                ...<options>         - 0을 사용하도록 권장하며, 그 이외의 설명은 없다.
                ...<outUsedScanMode> - RPLIDAR가 사용할 스캔모드 가ㅄ이 반환되는 변수.
            */
            RplidarScanMode scanMode;
            rplidarA1 -> startScan(false, true, 0, &scanMode);
        }
    
        // 연결이 실패하였으면 아래의 코드를 실행한다.
        else {fprintf(stderr, "Failed to connect to LIDAR %08x\r\n", result);}
        /*__________ [END]: RPLIDAR A1 센서 제어 관련 설정 __________*/
            
            
            
        // CREATE VECTOR OBJECT CALLED "detection1" WHICH CAN CONTAIN LIST OF MAT OBJECTS.
        /*
            >> `std::vector< std::vector< std::vector<Mat> > >`: Creates 3D vector array containing Mat data-type.
        */
        Mat frame, blob;
        cv::VideoCapture cap;

        // OPEN DEFAULT CAMERA OF `/dev/video0` WHERE ITS INTEGER IS FROM THE BACK.
        /*
            Set the video resolution by `cap` as designated pixel size, does not work on called video file.
            Access, or "open" default camera which is presumes to be `/dev/video0` file
                ...(which is where number 0 may have derived from).
        */

        cap.open(1); // 노트북 카메라는 cap.open(1) 또는 cap.open(-1)
        // USB 카메라는 cap.open(0);
        int x1;
        int y1;
        int x2;
        int y2;
        bool found = true;// FIXME INEFFICIENT CODE
        int countFrame = 0;
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

//        frontal_face_detector detector = get_frontal_face_detector();
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
//        shape_predictor pose_model;
//        deserialize("models/shape_predictor_68_face_landmarks.dat") >> pose_model;
        
        // ASSIGN VARIABLE "net" AS AN OBJECT OF "anet_type" DEFINED ABOVE.
        /*
            >> `models/dlib_face_recognition_resnet_model_v1.dat`: DNN for a "FACIAL RECOGNITION".
                ...it is presume this file too needs deserialization which reconstructs to original data format.
                ...for more information of a serialization, read this webpage; https://www.geeksforgeeks.org/serialization-in-java/.
            >> statement `dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;`
                ...reconstructed recognition model is placed in a hollow neural network frame manually created using operator `>>`.
                ...where now this variable `net` works as a model.
        */

//        anet_type net;
//        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;



//        Net net2 = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
//
//        net2.setPreferableTarget(DNN_TARGET_OPENCL);

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
//        dlib::array<matrix<rgb_pixel>> faces;

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

        cv::Mat user_img = cv::imread(userImg,cv::IMREAD_COLOR);

        std::vector<dlib::rectangle> locations = recognizer.faceDetection(user_img);
        dlib::array<matrix<rgb_pixel>> faces = recognizer.faceLandMark(user_img,locations);

        // CREATE A VARIALBE "face_detected_user" FOR FUTURE FACIAL COMPARISON.
        /*
            >> It is still unclear what the stored value `face_detected_user[0]` represents,
                ...but it is possible the value is (1) Loss, (2) Score, or (3) Confidence.
            >> `net_recognition(<input_data>,<batch_size>)`: uncertain of a purpose of a <batch_size> is; there's only one facial data here!
        */
        std::vector<matrix<float,0,1>> face_descriptors = recognizer.faceEncoding(faces);



        // __________ PROCESS OF PERSON DETECTION USING EXISTING FRAMEWORK MODEL. __________ //

        // CAFFE의 PROTOTXT와 CAFFEMODEL는 DARKNET의 CONFIG와 WEIGHT 파일과 동일하다 종류의 파일이다;
        // ...FOR PERSON DETECTION (NOT A FACIAL DETECTION)
//        String prototxt = "models/MobileNetSSD_deploy.prototxt";
//        String model = "models/MobileNetSSD_deploy.caffemodel";

        // 클래스 목록을 수집한다.
        /*
            >> `cv::String::c_str()`: "std::string::c_str()"로서도 함수가 존재;
                ...NULL(즉, 띄어쓰기 혹은 줄바꿈)이 있을 때마다 문자열을 나누어 행렬로 반환.
            >> `cv::vector::push_back(<data>)`: <data>를 현재 벡터의 맨 마지막으로 넣어준다.
                ...마치 스택 자료구조의 PUSH-BACK이라고 생각하면 된다.
                    결과적으로, 모든 클래스 종류는 "classes"라는 벡터 변수에 저장된다.
        */
//        String classesFile = "names";
//        ifstream ifs(classesFile.c_str());
//        string line;
//        std::vector<string> classes;
//        while(getline(ifs,line))
//            classes.push_back(line);

        // CAFFE의 설정파일과 모델파일을 OpenCV에서 미리 준비된 신경망으로 불러 넣어준다:
        // 이는 얼굴 탐지가 아닌 "사람 탐지"을 위해 사용된다.
//        Net net1 = readNetFromCaffe(prototxt, model);
//        net1.setPreferableTarget(DNN_TARGET_OPENCL);

        // 웹캠으로 촬영이 진행되는 동안...
        while(cap.isOpened()){
            
            /*__________ [START]: RPLIDAR A1 센서 제어: GKO95 작성 __________*/
                
            // 스캔 데이터인 노드(node)를 담을 수 있는 배열을 생성한다.
            rplidar_response_measurement_node_hq_t nodes[8192];

            // 노드 개수(8192)를 계산적으로 구한다.
            size_t nodeCount = sizeof(nodes)/sizeof(rplidar_response_measurement_node_hq_t);
            
            // 완전한 0-360도, 즉 한 사이클의 스캔이 완료되었으면 스캔 정보를 획득한다.
            /*
                >> `grabScanDataHq(<nodebuffer>,<count>)`: 본 API로 획득한 정보들은 항상 다음과 같은 특징을 가진다:

                    1) 획득한 데이터 행렬의 첫 번째 노드, 즉 <nodebuffer>[0]는 첫 번째 스캔 샘플값이다 (start_bit == 1).
                    2) 데이터 전체는 정확히 한 번의 360도 사이클에 대한 스캔 정보만을 지니고 있으며, 그 이상 혹은 그 이하도 아니다.
                    3) 각도 정보는 항상 오름차순으로 나열되어 있지 않다. 이는 ascendScanData API를 사용하여 오름차순으로 재배열 가능하다.

                    ...<nodebuffer> - API가 스캔 정보를 저장할 수 있는 버퍼.
                    ...<count>      - API가 버퍼에게 전달할 수 있는 최대 데이터 개수를 초기설정해야 한다.
                                    API의 동작이 끝났으면 해당 파라미터로 입력된 변수는 실제로 스캔된 정보 개수가 할당된다 (예. 8192 -> 545)
            */
            result = rplidarA1->grabScanDataHq(nodes, nodeCount);
            
            // 스캔을 성공하였을 경우 아래의 코드를 실행한다.
            if (IS_OK(result)) {    // START OF IF CONDITION: IF SCAN IS COMPLETE

                // <angleRange>: 총 방향 개수, <distances[]>: 거리를 담는 배열, <count>: 방향 카운터, <angleOF_prev>: 이전 위상가ㅄ을 받아내기 위한 변수.
                int angleRange = CYCLE/DIRECTION;
                int count = 0, angleOFF_prev = NULL;
                // 거리값을 계산하고 정리하기 위해 사용되는 임시 저장변수.
                int distancesTEMP[DIRECTION] = {0};

                // 순서를 오름차순으로 재배열한다.
                rplidarA1 -> ascendScanData(nodes, nodeCount);

                // 스캔 결과를 오름차순으로 하나씩 확인한다.
                for (int i = 0; i < nodeCount; i++){    // START OF FOR LOOP: READING SCAN DATA

                    // 각도는 도 단위 (+ 위상), 거리는 밀리미터 단위로 선정 (범위외 거리는 0으로 반환).
                    float angle = nodes[i].angle_z_q14 * 90.f / (1 << 14);
                    float distance = nodes[i].dist_mm_q2 / (1 << 2);
                    
                    // 위상 추가하여 방향성 교정.
                    angle = angle + angleRange/2;

                    // 하나의 방향이라고 인지할 수 있도록 정해놓은 batch 범위가 있으며, 중앙에서 얼마나 벗어난 각도인지 확인.
                    // 값이 크면 클수록 중앙과 가깝다는 의미.
                    int angleOFF = lround(angle) % angleRange;
                    angleOFF = abs(angleOFF - angleRange/2);
                    
                    // 현재 위상가ㅄ이 0이고 이전 위상가ㅄ이 1이면 방향 카운터를 증가시킨다.
                    // 반대로 설정하면 초반에 바로 (현재 = 1, 이전 = 0) 가ㅄ이 나올 수 있어 오류는 발생하지 않지만 첫 방향의 최소거리가 계산되지 않는다.
                    if (angleOFF == 0 && angleOFF_prev == 1) count++;
                    
                    // 처리되지 않은 나머지 전방 각도 당 거리를 계산하기 위하여 방향 카운터 초기화.
                    if (count == DIRECTION) count = 0;

                    // 루프를 돌기 전에 현재 위상가ㅄ을 이전 위상가ㅄ으로 할당한다.
                    angleOFF_prev = angleOFF;

                    // 0을 제외한 최소거리를 저장한다.
                    if (distancesTEMP[count] == 0) distancesTEMP[count] = distance;
                    else if (distancesTEMP[count] > distance && distance != 0) distancesTEMP[count] = distance;

                }   // END OF FOR LOOP: READING SCAN DATA.

                for (int i = 0; i < DIRECTION; i++) distances[i] = distancesTEMP[i];
                
            }   // END OF IF CONDITION: IF SCAN IS COMPLETE
            
            // FOR A SINGLE CYCLE
            // break;
            
            // 스캔을 실패하였을 경우 아래의 코드를 실행한다.
            if (IS_FAIL(result))
            {   
                std::cout << "[ERROR] FAILED TO SCAN USING LIDAR." << std::endl;
                break;
            }
            /*__________ [END]: RPLIDAR A1 센서 제어 __________*/
                
            // VIDEOCAPTURE 클래스의 "CAPTURE"는 촬영된 순간의 프레임을 cv::Mat 형태의 "FRAME" 오브젝트에 할당한다.
            cap >> frame;
            double t = cv::getTickCount();
            resize(frame,frame, Size(640,480));

            found = recognizer.humanDetection(frame);




            if(countFrame%3==0) {
                //face recoginition 구현 중
                if (found) {

                    char* data;

                    std::vector<dlib::rectangle> locations2 = recognizer.faceDetection(frame);


                    dlib::array<matrix<rgb_pixel>> faces2 = recognizer.faceLandMark(frame,locations2);

                    std::vector<matrix<float, 0, 1>> face_descriptors2 = recognizer.faceEncoding(faces2);
                    std::vector<String> names;
                    String name;

                    // START OF FOR LOOP: USER DETECTION AND LOCATION FINDER.
                    for (size_t i = 0; i < face_descriptors2.size(); ++i) {
                        name = "unknown";
                        if (length(face_descriptors[0] - face_descriptors2[i])< 0.5) {
                            name = "user";
                            long xcenter = (locations2[i].right() + locations2[i].left()) / 2;
                            long ycenter = (locations2[i].bottom() + locations2[i].top()) / 2;
                            long size = (locations2[i].right() - locations2[i].left());

                            cout << "size = " << size << endl;
                            //if(xcenter <100)
//                         {
//                             data="l";
//                             //cout<<"data = "<<data<<endl;

//                         }
//                         else if(xcenter > 520)
//                         {
//                             data="r";
//                             //cout<<"data = "<<data<<endl;

//                         }
                            //write(fd, data, 1);

                            //if (xcenter!=320&&ycenter!= 240)
                            //{
                            //    cout<<"중심을 벗어낫습니다"<<endl;
                            //}

                            if (tempsize == 0) {
                                tempsize = size;
                                std::cout << "값을 저장하였습니다" << endl;
                            } else {
                                if (tempsize < size - 5) {
                                    //cout<<"저장값 : "<<tempsize<<endl;
                                    //cout<<"data = "<<data<<endl;
                                    tempsize = size;
                                    data = "b";

                                    //data= "g";
                                    //write(fd, data, strlen(data));

                                } else if (tempsize > size + 5) {
                                    cout << "저장값 : " << tempsize << endl;
                                    //cout<<"data = "<<data<<endl;
                                    tempsize = size;
                                    data = "g";

                                    //data = "b";
                                    //write(fd, data, strlen(data));

                                } else {
                                    data = "s";
                                    //cout<<"data = "<<data<<endl;

                                }
                            }
                        }   // END OF FOR LOOP: USER DETECTION AND LOCATION FINDER.

                        /*__________ RPLIDAR 행동교정 함수 __________*/
                        data = rplidarBehavior(data, distances);
                        write(fd, data, strlen(data));

                        //cout<<"data = "<<data<<endl;
                        names.push_back(name);
                    }   // END OF IF CONDITION.

                    int i = 0;


                    for (int i = 0; i < locations2.size(); i++) {
                        cv::rectangle(frame, Point(locations2[i].left(), locations2[i].top()),
                                      Point(locations2[i].right(), locations2[i].bottom()), Scalar(0, 255, 0), 2);
                        putText(frame, names[i], Point(locations2[i].left() + 6, locations2[i].top() - 6),
                                FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 255, 255), 2);
                    }

//                    delete(&faces2);
//                    delete(&locations2);
                }


            }
            double tt_opencvDNN = 0;
            double fpsOpencvDNN = 0;



            tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fpsOpencvDNN = 1/tt_opencvDNN;
            putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

            // 웹캠에서 촬영하는 영상을 보여준다; Enter 키를 누르면 종료.
            cv::imshow("HumanTrackingUV",frame);
            if (cv::waitKey(30)==13) break;
            countFrame++;

        }// END OF WHILE LOOP

//        delete(&locations);
//        delete(&faces);
    }// END OF TRY BLOCK: WHOLE PROCESS FOR DETECTION AND AUTOMATION.

    // 예외처리 1: 랜드마크 마크 모델을 찾을 수 없습니다.
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    
    // 예외처리 2
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
        
    // 영상인식과 자율주행이 모두 끝나면 R/W 파일을 닫는다.
    close(fd);
        
    // 영상인식과 자율주행이 모두 끝났으면 OpenCV 창을 닫는다.
    cv::destroyWindow("HumanTrackingUV");
    
    // RPLIDAR 모터 중지.
    // 드라이버의 장치 연결을 끊는다.
    // RPLIDAR A1과 통신을 위한 장치 드라이버 제거.
    rplidarA1 -> stopMotor();
    rplidarA1 -> disconnect();
    RPlidarDriver::DisposeDriver(rplidarA1);
        
    return 0;
}


/*__________ RPLIDAR 행동교정 함수 정의: GKO95 작성 __________*/
char* rplidarBehavior(/*char detectPosition, */char* platformMove, int *distanceRPLIDAR) {

    // REFERENCE
    /*
        >> detectPostion: 영상에 탐지된 대상자가 왼쪽(l) 혹은 오른쪽(r)에 있는지 알려주는 파라미터.
        >> platformMove: 영상에 탐지된 대상자를 기반으로 전진(g), 후진(b), 좌회전(l), 우회전(r), 혹은 정지(s)하는지 알려주는 파라미터.
        >> distanceRPLIDAR = 전방으로 시작으로 시계방향으로 거리를 알려주는 파라미터; {전방, 우, 우X2, ..., 우X(n-1), 후방, 좌X(n-1),  ... , 좌X2, 좌}. 0은 측정범위 밖.
    */

    // 장애물 기준 거리를 300mm, 즉 0.3미터로 잡는다.
    #define DIST_STOP 300

    // 방향을 틀었을 때, 최소한 0.5미터의 여유가 있을 때로 선택한다.
    #define DIST_REF 500

    // 정지 신호에는 무조건 정지한다.
    if (platformMove == STOP) return STOP;

    // 전방에 장애물이 존재할 경우 (0은 측정범위 밖); 후진과 정지는 따로 조건문이 주어져 있으므로 고려하지 않는다.
    if (0 < *(distanceRPLIDAR + DIRECTION/2) && *(distanceRPLIDAR + DIRECTION/2) <= DIST_STOP && platformMove != STOP && platformMove != BACK){

        // 전 방향의 거리여부를 앞에서부터 뒤로 좌우를 동시에 확인한다 (후방 제외).
        for (int i = (DIRECTION/2) -1; i > 0; i--){

            // 오른쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 왼쪽보다 거리적 여유가 많을 시 오른쪽으로 회전한다.
            if ((*(distanceRPLIDAR + i) > DIST_REF && *(distanceRPLIDAR + i) >= *(distanceRPLIDAR + (DIRECTION - i)) && *(distanceRPLIDAR + (DIRECTION - i)) != 0) || *(distanceRPLIDAR + i) == 0)
                return LEFT;
            // 반면 왼쪽이 정해진 기준보다 거리적 여유가 있는 동시에, 오른쪽보다 거리적 여유가 많을 시에는 왼쪽으로 회전한다.
            else if((*(distanceRPLIDAR + (DIRECTION - i)) > DIST_REF  && *(distanceRPLIDAR + i) <= *(distanceRPLIDAR + (DIRECTION - i)) &&  *(distanceRPLIDAR + i) != 0 ) || *(distanceRPLIDAR + (DIRECTION - i)) == 0 )
                return RIGHT;
        }

        // 위의 조건문을 만족하지 않았다는 것은 정해진 기준의 여유보다 거리가 적다는 의미이다.

        // 후방 거리여부를 확인하고, 전방향이 막혀 있으면 움직이지 않는다.
        if (*(distanceRPLIDAR) > DIST_REF) return BACK;
        else return STOP;
    }

    // 뒤에 장애물이 있으면 뒤로 움직이는 신호에도 정지시킨다.
    if (platformMove == BACK && *(distanceRPLIDAR) <= DIST_REF) return STOP;
    
    // 아무런 조건을 만족하지 않으면 장애물의 구애를 받지 않는다는 의미로 신호대로 움직인다.
    return platformMove;
}



