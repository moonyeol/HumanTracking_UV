//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고
//
//컴파일 방
//g++ -std=c++11 -O3 -I.. /home/nvidia/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o 1119 socketnonclass_1.cpp $(pkg-config opencv4 --libs --cflags)
#include <opencv2/videoio.hpp>


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dict.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>    /* Standard input/output definitions */
#include <string>
#include <string.h>   /* String function definitions */
#include <unistd.h>   /* UNIX standard function definitions */
#include <fcntl.h>    /* File control definitions */
#include <termios.h>  /* POSIX terminal control definitions */
#include <thread>

/*__________ RPLIDAR __________*/
//#include <rplidar.h>
//#include "rplidar.hpp"
//#include <cmath>
/*socket*/
#include "socket.h"

using namespace cv;
using namespace cv::dnn;
using namespace dlib;
using namespace std;
//using namespace rp::standalone::rplidar;


#define CYCLE 360       // 한 사이클은 360도.
#define DIRECTION 4     // 이동방향 개수.

#define LEFT 'l'
#define RIGHT 'r'
#define GO 'g'
#define BACK 'b'
#define STOP 's'


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
// CAFFE의 PROTOTXT와 CAFFEMODEL는 DARKNET의 CONFIG와 WEIGHT 파일과 동일하다 종류의 파일이다;
long tempsize=0;
char data = STOP;
bool isEnd = false;
bool isStart = false;

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

    void readClasses(){
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

    std::vector<matrix<float,0,1>> faceEncoding(dlib::array<matrix<rgb_pixel>>& faces){

        return faceEncoder(faces,16);
    }

//    float calDistance(matrix<float, 0, 1> descriptors1, matrix<float, 0, 1> descriptors2){
//        return length(descriptors1-descriptors2);
//    }

};


void humanTracking(){






    try {   // TRY BLOCK CODE START: WHOLE PROCESS FOR DETECTION AND AUTOMATION.
	while(true){
		cout << "";
		if(isStart)break;
		continue;}
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





        // 웹캠으로 촬영이 진행되는 동안...
        while(cap.isOpened()){




            // VIDEOCAPTURE 클래스의 "CAPTURE"는 촬영된 순간의 프레임을 cv::Mat 형태의 "FRAME" 오브젝트에 할당한다.
            cap >> frame;
            double t = cv::getTickCount();
            resize(frame,frame, Size(640,480));
	    found = false;
            //found = recognizer.humanDetection(frame);




            if(countFrame%3==0) {   // START OF OUTER IF CONDITION
                //face recoginition 구현 중
                //if (found) {    // START OF INNER IF CONDITION.



                    std::vector<dlib::rectangle> locations2 = recognizer.faceDetection(frame);


                    dlib::array<matrix<rgb_pixel>> faces2 = recognizer.faceLandMark(frame,locations2);

                    std::vector<matrix<float, 0, 1>> face_descriptors2 = recognizer.faceEncoding(faces2);
                    std::vector<String> names;
                    String name;

                    // START OF FOR LOOP: USER DETECTION AND LOCATION FINDER.
                    for (size_t i = 0; i < face_descriptors2.size(); ++i) {
                        name = "unknown";
                        if (length(face_descriptors[0] - face_descriptors2[i])< 0.5) {found = true;
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
                            cout <<"data(main) = "<< data<<endl;
                        }   // END OF FOR LOOP: USER DETECTION AND LOCATION FINDER.


                        //cout<<"data = "<<data<<endl;
                        names.push_back(name);

                    }   // END OF INNER IF CONDITION.
			if(!found){data = STOP;}
ofstream camera_data("from_opencv");
		if(camera_data.is_open())
{camera_data << data;}
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
            fpsOpencvDNN = 1000/tt_opencvDNN;
            putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

            // 웹캠에서 촬영하는 영상을 보여준다; Enter 키를 누르면 종료.
            cv::imshow("HumanTrackingUV",frame);
           if (cv::waitKey(60)==13) break;
	if(isEnd) break;

            countFrame++;


        }// END OF WHILE LOOP
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


}
/*void lidar(int fd) {
		while(true){
		cout << "";
		if(isStart)break;
		continue;}
    class rplidar rplidarA1;
    char *move;
    while (!isEnd) {
        rplidarA1.scan();
        move = rplidarA1.move(data);
        rplidarA1.result();
        write(fd, move, strlen(move));
//	if(IS_FAIL(rplidarA1.RESULT)) {
//	    rplidarA1.~rplidar();
//	    exit(EXIT_FAILURE);
//	}
    }
}
*/
void socketFunc(){
    std::string a("end\n");
    std::string b("start\n");
    char *socket_data;

    char readBuff[BUFFER_SIZE];
    char sendBuff[BUFFER_SIZE];
    struct sockaddr_in serverAddress, clientAddress;
    int server_fd, client_fd;
    unsigned int client_addr_size;
    ssize_t receivedBytes;
    ssize_t sentBytes;

    socklen_t clientAddressLength = 0;
 
    memset(&serverAddress, 0, sizeof(serverAddress));
    memset(&clientAddress, 0, sizeof(clientAddress));
 
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddress.sin_port = htons(20162);
 
 
    // 서버 소켓 생성 및 서버 주소와 bind
 
    // 서버 소켓 생성(UDP니 SOCK_DGRAM이용)
    if ((server_fd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) // SOCK_DGRAM : UDP
    {
        printf("Sever : can not Open Socket\n");
        exit(0);
    }
 
    // bind 과정
    if (bind(server_fd, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0)
    {
        printf("Server : can not bind local address");
        exit(0);
    }
 
 
    printf("Server: waiting connection request.\n");
     while (1)
    {
 	
      
        // 클라이언트 IP 확인
        struct sockaddr_in connectSocket;
        socklen_t connectSocketLength = sizeof(connectSocket);
        getpeername(client_fd, (struct sockaddr*)&clientAddress, &connectSocketLength);
        //char clientIP[sizeof(clientAddress.sin_addr) + 1] = { 0 };
        //sprintf(clientIP, "%s", inet_ntoa(clientAddress.sin_addr));
        // 접속이 안되었을 때는 while에서 출력 x
        //if(strcmp(clientIP,"0.0.0.0") != 0)
            //printf("Client : %s\n", clientIP);

 
 
        //채팅 프로그램 제작
        client_addr_size = sizeof(clientAddress);
 
        receivedBytes = recvfrom(server_fd, readBuff, BUFF_SIZE, 0, (struct sockaddr*)&clientAddress, &client_addr_size);
	socket_data = readBuff;
	//printf("%s \n",socket_data);

	


        readBuff[receivedBytes] = '\0';
	//printf("%s",readBuff);
	
        //fputs(readBuff, stdout);
        fflush(stdout);
	
        sprintf(sendBuff, "%s", readBuff);


	sentBytes = sendto(server_fd, sendBuff, strlen(sendBuff), 0, (struct sockaddr*)&clientAddress, sizeof(clientAddress));
	if(a.compare(sendBuff) == 0){
		cout<<"프로세스를 멈춥니다."<<endl;
		isEnd = true;
    		break;
	}
	else if(b.compare(sendBuff) == 0){
		cout<<"프로세스를 시작합니다"<<endl;
		isStart = true;
	} 
    }
    close(server_fd);
}


int main(int argc, char **argv ) {


    
 

		thread sThread(socketFunc);
     		thread hThread(humanTracking);
    		//thread lThread{lidar,fd};

    		hThread.join();
		sThread.join();
    		//lThread.join();
	




   






    // 영상인식과 자율주행이 모두 끝나면 R/W 파일을 닫는다.
    //close(fd);

    // 영상인식과 자율주행이 모두 끝났으면 OpenCV 창을 닫는다.
    cv::destroyWindow("HumanTrackingUV");

    return 0;
}
