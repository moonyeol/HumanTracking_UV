//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고
//
//컴파일 방
//g++ -std=c++11 -O3 -I.. /home/e-on/dlib/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o test2 test2.cpp $(pkg-config opencv4 --libs --cflags)


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

#define tensor

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./models/opencv_face_detector_uint8.pb";









int main()
{

    try
    {
//        cv::cuda::GpuMat frame, blob, grayFrame;
        Mat frame, blob, grayFrame;
        std::vector<Mat> detection1;
        std::vector<std::vector<std::vector<Mat>>> detection2;
        cv::VideoCapture cap;
        cap.open(0); // 노트북 카메라는 cap.open(1) 또는 cap.open(-1)
        // USB 카메라는 cap.open(0);
        int x1;
        int y1;
        int x2;
        int y2;
        bool found = false;

        // Load face detection and pose estimation models.






        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

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
        dlib::array<matrix<rgb_pixel>> faces;


      //face recoginition 구현 중

        matrix<rgb_pixel> user_img;
        load_image(user_img, "user.jpg");

        for (auto face : detector(user_img)) {
            auto shape = pose_model(user_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(user_img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }
        std::vector<matrix<float,0,1>> face_descriptors = net(faces,16);




        String prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt";
        String model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel";
        String classesFile = "names";
        ifstream ifs(classesFile.c_str());
        string line;
        std::vector<string> classes;
        while(getline(ifs,line))
            classes.push_back(line);


        Net net1 = readNetFromCaffe(prototxt, model);
        net1.setPreferableTarget(DNN_TARGET_OPENCL);

        // Grab and process frames until the main window is closed by the user.
        while(cap.isOpened()){
            // Grab a frame
            cap >> frame;
            double t = cv::getTickCount();
            resize(frame,frame, Size(640,480));

            blob = blobFromImage(frame, 0.007843, Size(300,300), 127.5);
            net1.setInput(blob);

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

                matrix<rgb_pixel> img;
//                cv::cuda::GpuMat rgb_frame;
                Mat rgb_frame;
                cvtColor(frame,rgb_frame,COLOR_BGR2RGB);

                dlib::assign_image(img, dlib::cv_image<rgb_pixel>(rgb_frame));




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
                    }

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



            cv::imshow("EXAMPLE02",frame);
            if (cv::waitKey(30)==27)
            {
                break;
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
    cv::destroyWindow("EXAMPLE02");
    return 0;
}

