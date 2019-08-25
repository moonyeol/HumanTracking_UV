//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고


// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/


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


std::vector<String> getOutputsNames(const Net& net);



//
//class face_recognition_model_v1
//{
//
//public:
//
//    face_recognition_model_v1(const std::string& model_filename)
//    {
//        deserialize(model_filename) >> net;
//    }
//
//    matrix<double,0,1> compute_face_descriptor (
//            matrix<rgb_pixel> img,
//            const full_object_detection& face,
//            const int num_jitters,
//            float padding = 0.25
//    )
//    {
//        std::vector<full_object_detection> faces(1, face);
//        return compute_face_descriptors(img, faces, num_jitters, padding)[0];
//    }
//
//    matrix<double,0,1> compute_face_descriptor_from_aligned_image (
//            matrix<rgb_pixel> img,
//            const int num_jitters
//    )
//    {
//        std::vector<matrix<rgb_pixel>> images{img};
//        return batch_compute_face_descriptors_from_aligned_images(images, num_jitters)[0];
//    }
//
//    std::vector<matrix<double,0,1>> compute_face_descriptors (
//            matrix<rgb_pixel> img,
//            const std::vector<full_object_detection>& faces,
//            const int num_jitters,
//            float padding = 0.25
//    )
//    {
//        std::vector<matrix<rgb_pixel>> batch_img(1, img);
//        std::vector<std::vector<full_object_detection>> batch_faces(1, faces);
//        return batch_compute_face_descriptors(batch_img, batch_faces, num_jitters, padding)[0];
//    }
//
//    std::vector<std::vector<matrix<double,0,1>>> batch_compute_face_descriptors (
//            const std::vector<matrix<rgb_pixel>>& batch_imgs,
//    const std::vector<std::vector<full_object_detection>>& batch_faces,
//    const int num_jitters,
//    float padding = 0.25
//    )
//    {
//
//        if (batch_imgs.size() != batch_faces.size())
//            throw dlib::error("The array of images and the array of array of locations must be of the same size");
//
//        int total_chips = 0;
//        for (const auto& faces : batch_faces)
//        {
//            total_chips += faces.size();
//            for (const auto& f : faces)
//            {
//                if (f.num_parts() != 68 && f.num_parts() != 5)
//                    throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style or dlib's 5 point style.");
//            }
//        }
//
//
//        dlib::array<matrix<rgb_pixel>> face_chips;
//        for (int i = 0; i < batch_imgs.size(); ++i)
//        {
//            auto& faces = batch_faces[i];
//            auto& img = batch_imgs[i];
//
//            std::vector<chip_details> dets;
//            for (const auto& f : faces)
//                dets.push_back(get_face_chip_details(f, 150, padding));
//            dlib::array<matrix<rgb_pixel>> this_img_face_chips;
//            extract_image_chips(img, dets, this_img_face_chips);
//
//            for (auto& chip : this_img_face_chips)
//                face_chips.push_back(chip);
//        }
//
//        std::vector<std::vector<matrix<double,0,1>>> face_descriptors(batch_imgs.size());
//        if (num_jitters <= 1)
//        {
//            // extract descriptors and convert from float vectors to double vectors
//            auto descriptors = net(face_chips, 16);
//            auto next = std::begin(descriptors);
//            for (int i = 0; i < batch_faces.size(); ++i)
//            {
//                for (int j = 0; j < batch_faces[i].size(); ++j)
//                {
//                    face_descriptors[i].push_back(matrix_cast<double>(*next++));
//                }
//            }
//            DLIB_ASSERT(next == std::end(descriptors));
//        }
//        else
//        {
//            // extract descriptors and convert from float vectors to double vectors
//            auto fimg = std::begin(face_chips);
//            for (int i = 0; i < batch_faces.size(); ++i)
//            {
//                for (int j = 0; j < batch_faces[i].size(); ++j)
//                {
//                    auto& r = mean(mat(net(jitter_image(*fimg++, num_jitters), 16)));
//                    face_descriptors[i].push_back(matrix_cast<double>(r));
//                }
//            }
//            DLIB_ASSERT(fimg == std::end(face_chips));
//        }
//
//        return face_descriptors;
//    }
//
//    std::vector<matrix<double,0,1>> batch_compute_face_descriptors_from_aligned_images (
//            const std::vector<matrix<rgb_pixel>>& batch_imgs,
//    const int num_jitters
//    )
//    {
//        dlib::array<matrix<rgb_pixel>> face_chips;
//        for (auto& img : batch_imgs) {
//
//            matrix<rgb_pixel> image;
//
//                assign_image(image, matrix<rgb_pixel>(img));
//
//
//            // Check for the size of the image
//            if ((image.nr() != 150) || (image.nc() != 150)) {
//                throw dlib::error("Unsupported image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it. \
//                That is, centered and scaled essentially the same way.");
//            }
//
//            face_chips.push_back(image);
//        }
//
//        std::vector<matrix<double,0,1>> face_descriptors;
//        if (num_jitters <= 1)
//        {
//            // extract descriptors and convert from float vectors to double vectors
//            auto descriptors = net(face_chips, 16);
//
//            for (auto& des: descriptors) {
//                face_descriptors.push_back(matrix_cast<double>(des));
//            }
//        }
//        else
//        {
//            // extract descriptors and convert from float vectors to double vectors
//            for (auto& fimg : face_chips) {
//                auto& r = mean(mat(net(jitter_image(fimg, num_jitters), 16)));
//                face_descriptors.push_back(matrix_cast<double>(r));
//            }
//        }
//        return face_descriptors;
//    }
//
//private:
//
//    dlib::rand rnd;
//
//    std::vector<matrix<rgb_pixel>> jitter_image(
//            const matrix<rgb_pixel>& img,
//            const int num_jitters
//    )
//    {
//        std::vector<matrix<rgb_pixel>> crops;
//        for (int i = 0; i < num_jitters; ++i)
//            crops.push_back(dlib::jitter_image(img,rnd));
//        return crops;
//    }
//
//};



int main()
{
    // TRY BLOCK CODE START
    try
    {
        cv::Mat frame, blob;
        
        // CREATE VECTOR OBJECT CALLED "detection1" WHICH CAN CONTAIN LIST OF MAT OBJECTS.
        /*
            >> `std::vector< std::vector< std::vector<Mat> > >`: Creates 3D vector array containing Mat data-type.
        */
        std::vector<cv::Mat> detection1;
        std::vector<std::vector<std::vector<cv::Mat>>> detection2;
        cv::VideoCapture cap;
        
        // OPEN DEFAULT CAMERA OF `/dev/video0` WHERE ITS INTEGER IS FROM THE BACK.
        cap.open(0);
        int x1;
        int y1;
        int x2;
        int y2;


        // DECLARE A FRONTAL FACE DETECTOR (HOG BASED) TO A VARIABLE "detector". 
        /*
            >> `dlib::get_frontal_face_detector()`: return "dlib::rectangles" object upon detecting a frontal face.
                ...and to assign the detector as a variable needs data-type which is `dlib::frontal_face_detector`.
                ...Python didn't needed this data-type since variable in Python does not need to data-type designation.
        */
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        
        // FACIAL LANDMARK DETECTION
        /*
            >> `dlib::shape_predictor`: returns set of point locations that define the pose of the object.

            >> `dlib::deserialize()`  : recover data saved in serialized back to its original format with deserialization.
                ...the operator `>>` imports the file of model to the "dlib::shape_predictor" object.

            >> `shape_predictor_68_face_landmarks.dat`: a file containing a model of pin-pointing 68 facial landmark.
                ...data is saved in serialized format which can be stored, transmitted and reconstructed (deserialization) later.
                ...얼굴 랜드마크 데이터 파일 "shape_predictor_68_face_landmarks.dat"의 라이선스는 함부로 상업적 이용을 금합니다.
                ...본 파일을 상업적 용도로 사용하기 위해서는 반드시 Imperial College London에 연락하기 바랍니다.
        */
        dlib::shape_predictor pose_model;
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // ASSIGN VARIABLE "net" AS AN OBJECT OF "anet_type" DEFINED ABOVE.
        /*
            >> `models/dlib_face_recognition_resnet_model_v1.dat`: DNN for a facial recognition.
                ...it is presume this file too needs deserialization which reconstructs to original data format.
                ...for more information of a serialization, read this webpage; https://www.geeksforgeeks.org/serialization-in-java/.
        */
        anet_type net;
        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;

        //face recoginition 구현 중
        /*
            >> `dlib::array`    :
            >> `dlib::matrix`   :
            >> `dlib::rgb_pixle`:
        */
        dlib::array<dlib::matrix<dlib::rgb_pixel>> face_chips;
        
        //
        /*
            >> `std::vector`: similar to `std::array` but stores data in heap (thus, always calls `new`) and is resizable.
                ...Hence, for a big amount of data, it is recommended to use `std::vector` than `std::array`.
        */
        std::vector<dlib::matrix<dlib::rgb_pixel>> face_locations;
        
        cv::Mat user_img = imread("user.jpg");
        for (auto face : detector(user_img)){
            auto shape = pose_model(user_img, face);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            extract_image_chip(user_img, get_face_chip_details(shape,150,0.25), face_chip);
            face_locations.push_back(move(face_chip));
            // Also put some boxes on the faces so we can see that the detector is finding
            // them.
        }
        if (face_locations.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            return 1;
        }

        
        std::vector<matrix<float,0,1>> face_descriptors = net(face_locations);

        auto face_encoding = face_descriptors[0];

        // PROTOTXT AND CAFFEMODEL IS A COUNTERPART OF CONFIG AND WEIGHT IN DARKNET.
        cv::String prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt";
        cv::String model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel";
        
        // ACQUIRE LIST OF CLASSES.
        /*
            >> `cv::String::c_str()`: also available as "std::string::c_str()";
                ...returns array with strings splitted on NULL (blank space, new line). 
            
            >> `cv::vector::push_back(<data>)`: push the data at the back end of the current last element.
                ...just like a push-back of the stack data structure.

                    Hence, the name of the classes are all stored in variable "classes".
        */
        String classesFile = "names";
        std::ifstream ifs(classesFile.c_str());
        std::string line;
        std::vector<std::string> classes;
        while(getline(ifs,line))
            classes.push_back(line);

        // IMPORT CAFFE CONFIG AND MODEL TO THE NEURAL NETWORK.
        Net net1 = readNetFromCaffe(prototxt, model);

        // Grab and process frames until the main window is closed by the user.
        while(cap.isOpened()){
            // VIDEOCAPTURE "CAPTURE" RETURN ITS FRAME TO MAT "FRAME" IN 600x600 PIXEL.
            cap >> frame;
            resize(frame,frame, Size(600,600));
            
            double w = frame.cols;
            double h = frame.rows;
            
            // CONVERT FRAME TO PREPROCESSED "BLOB" FOR MODEL PREDICTION.
            /*
                >> `blobFromImage(<input>, <output>, <scalefactor>, <size>, <mean>, <swapRB>, <crop>, <ddepth>)`
                    ...scalefactor of (1/255) would normalize RGB value from (0~255) to (0~1).
                    ...cv::Size(224,224): advise to resize the output blob to 224x224 (GoogLeNet) to be passed through the model.
            */
            blob = blobFromImage(frame, 0.007843, cv::Size(frame.cols , frame.rows), 127.5);
            net1.setInput(blob);

            // RUN FORWARD PASS FOR THE WHOLE NETWORK: forward() [1/4]
            /*
                >> Return cv::Mat type outputBlobs variable "detection".
                    ...inside is a blob for first output of specified layer.
            */
            cv::Mat detection = net1.forward();


            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect> boxes;
            // for(size_t i=0; detection.size.p.size.p; i++) {
            
                //
                /*
                    >> `cv::Vec4i::Vec(<int>,<int>,<int>,<int>)`: 
                    >> `cv::Mat::at()`: 
                */
                cout << "A.channels()= " << detection.at<float>(Vec<int,4>(0,0,0,0)) << endl;
                // cout << "A.rows, A.cols = " << detection[0][0][i].rows << ", " << detection[0][0][i].cols << endl << endl;
                // cout << "A = " << detection[0][0][i] << endl << endl;
                // }
                cout << "size " <<detection.size.p[2]<<endl;
            cout << "size2 " <<detection.size.p[3]<<endl;
            for(int i =0; i < detection.size.p[2]; i++){
                float confidence = detection.at<float>(Vec<int,4>(0,0,i,2));
                    cout << confidence << endl;
                    if(confidence > 0.6){
                        int idx = detection.at<float>(Vec<int,4>(0,0,i,1));
                        String label = classes[idx];
                        cout << "test" << endl;
                        if(label.compare("person"))
                            continue;
                int startX = (int) (detection.at<float>(Vec<int,4>(0,0,i,3)) * frame.cols);
                int startY = (int) (detection.at<float>(Vec<int,4>(0,0,i,4)) * frame.rows);
                int endX = (int) (detection.at<float>(Vec<int,4>(0,0,i,5)) * frame.cols);
                int endY = (int) (detection.at<float>(Vec<int,4>(0,0,i,6)) * frame.rows);
                        cout << "startX" << startX <<endl;
                        cout << "startY" << startY <<endl;
                        cout << "endX" << endX <<endl;
                        cout << "endY" << endY <<endl;
                        cv::rectangle(frame, Point(startX,startY), Point(endX,endY),Scalar(0, 255, 0),2);
                        putText(frame, label, Point(startX, startY-15), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0,255,0),2);
                    }
            }






            //face recoginition 구현 중
            dlib::array<matrix<rgb_pixel>> face_chips1;
            std::vector<matrix<rgb_pixel>> face_locations1;
            auto locations = detector(frame);
            for (auto face : locations){
                auto shape = pose_model(frame, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(frame, get_face_chip_details(shape,150,0.25), face_chip);
                face_locations1.push_back(move(face_chip));
                // Also put some boxes on the faces so we can see that the detector is finding
                // them.
            }


            //
            /*
                >> `dlib::matrix<datatype, 0L, 1L>`: 0L and 1L is a size of column and row of matrix.
                    ...dlib::matrix<float, 0, 1> means it's a column vector which will be sized on runtime.

                >> Variable "f_names": 
            */
            std::vector<dlib::matrix<float,0,1>> face_descriptors1 = net(face_locations1);
            std::vector<cv::String> f_names;

            //
            for(auto f : face_descriptors1){
                auto distance = face_encoding - f;
                auto min_value = min(distance);
                cv::String name = "Unknown";
                if(min_value < 0.6){
                    name = "User";
                }
                f_names.push_back(name);
            }
            
            // DRAW RECTANGLE AROUND THE DETECTED OBJECT AND SHOW TEXT OF WHAT CLASS IT IS.
            for(int i =0; i < locations.size(); i++){
                cv::rectangle(frame, Point(locations[i].left(),locations[i].top()), Point(locations[i].right(),locations[i].bottom()),Scalar(0, 255, 0),2);
                putText(frame, f_names[i], Point(locations[i].left() +6, locations[i].top()-6), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255,255,255),2);

            }


//           for(size_t i = 0; i < detection.size.p[2]; i++){
//                float* data = (float*)detection[i].data;
//                for (int j = 0; j < detection[i].rows; ++j, data += detection[i].cols) {
//                    Mat scores = detection[i].row(j).colRange(5, detection[i].cols);
//                    Point classIdPoint;
//                    double confidence;
//                    // Get the value and location of the maximum score
//                    minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
//                    if (confidence > 0.5) {
//                        int centerX = (int) (data[0] * frame.cols);
//                        int centerY = (int) (data[1] * frame.rows);
//                        int width = (int) (data[2] * frame.cols);
//                        int height = (int) (data[3] * frame.rows);
//                        int left = centerX - width / 2;
//                        int top = centerY - height / 2;
//
//                        classIds.push_back(classIdPoint.x);
//                        confidences.push_back((float) confidence);
//                        boxes.push_back(Rect(left, top, width, height));
//                    }
//                }
//            }
//            std::vector<int> indices;
//            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
//            for (size_t i = 0; i < indices.size(); ++i)
//            {
//                int idx = indices[i];
//                Rect box = boxes[idx];
//
//                cv::rectangle(frame, Point(box.x,box.y),Point(box.x+box.width, box.y+box.height), Scalar(0, 255, 0), 2);
//                string label = format("%.2f", confidences[idx]);
//                if (!classes.empty())
//                {
//                    CV_Assert(classIds[idx] < (int)classes.size());
//                    label = classes[classIds[idx]] + ":" + label;
//                }
//                int baseLine;
//                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//                int top = max(box.y, labelSize.height);
//                cv::rectangle(frame, Point(box.x, top - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
//                putText(frame, label, Point(box.x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
//            }






/*

//            if (!cap.read(frame))
//            {
//                break;
//            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<rgb_pixel> cimg(frame);

            // Detect faces
            std::vector<dlib::rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;

//            for(auto face: faces){
//                auto shape = pose_model(cimg, face);
//
//
//            }

            for (unsigned long i = 0; i < faces.size(); ++i) {
//                shapes.push_back(pose_model(cimg, faces[i]));
                x1 = faces[i].left();
                y1 = faces[i].top();
                x2 = faces[i].right();
                y2 = faces[i].bottom();
                cv::rectangle(frame, CvPoint(x1,y1),CvPoint(x2,y2), cv::Scalar(0,255,0), (int)(frame.rows/150.0), 4);
            }

            // Display it all on the screen
//            win.clear_overlay();
//            win.set_image(cimg);
//            win.add_overlay(render_face_detections(shapes));
*/
            
            // SHOW CAPTURED VIDEO.
            cv::imshow("EXAMPLE02",frame);
            if (cv::waitKey(30)==27)
            {
                break;
            }
        }   // END OF WHILE LOOP
    }   // END OF TRY BLOCK
    
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
    cv::destroyWindow("EXAMPLE02");
    return 0;
}



// Get the names of the output layers
std::vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}
