//https://github.com/davisking/dlib/blob/master/tools/python/src/face_recognition.cpp
//http://dlib.net/dnn_face_recognition_ex.cpp.html
//기존 face_recognition_dlib
//위 3개 참고
//
//컴파일 방
//g++ -std=c++11 -O3 -I.. /home/e-on/dlib/dlib/dlib/all/source.cpp -lpthread -lX11 -ljpeg -DDLIB_JPEG_SUPPORT -o test2 test2.cpp $(pkg-config opencv4 --libs --cflags)
//









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

    try
    {
        Mat frame, blob;
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

      //face recoginition 구현 중

        matrix<rgb_pixel> user_img;
        load_image(user_img, "user.jpg");

        dlib::array<matrix<rgb_pixel>> faces;
        for (auto face : detector(user_img)) {
            auto shape = pose_model(user_img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(user_img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
        }
        std::vector<matrix<float,0,1>> face_descriptors = net(faces,16);



//        for (auto face : detector(user_img)){
//            auto shape = pose_model(user_img, face);
//            matrix<rgb_pixel> face_chip;
//            extract_image_chip(user_img, get_face_chip_details(shape,150,0.25), face_chip);
//            face_locations.push_back(move(face_chip));
//            // Also put some boxes on the faces so we can see that the detector is finding
//            // them.
//        }
//        if (face_locations.size() == 0)
//        {
//            cout << "No faces found in image!" << endl;
//            return 1;
//        }





//        std::vector<matrix<float,0,1>> face_descriptors = net(face_locations);

//        auto face_encoding = face_descriptors[0];



        String prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt";
        String model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel";
        String classesFile = "names";
        ifstream ifs(classesFile.c_str());
        string line;
        std::vector<string> classes;
        while(getline(ifs,line))
            classes.push_back(line);


        Net net1 = readNetFromCaffe(prototxt, model);

        // Grab and process frames until the main window is closed by the user.
        while(cap.isOpened()){
            // Grab a frame
            cap >> frame;
            resize(frame,frame, Size(600,600));
            double w = frame.cols;
            double h = frame.rows;
            blob = blobFromImage(frame, 0.007843, Size(frame.cols , frame.rows), 127.5);
            net1.setInput(blob);

            Mat detection = net1.forward();

            found =true;
            std::vector<int> classIds;
            std::vector<float> confidences;
            std::vector<Rect> boxes;
//            for(size_t i=0; detection.size.p.size.p; i++) {
//                cout << "A.channels()= " << detection.at<float>(Vec<int,4>(0,0,0,0)) << endl;
//                cout << "A.rows, A.cols = " << detection[0][0][i].rows << ", " << detection[0][0][i].cols << endl << endl;
//                cout << "A = " << detection[0][0][i] << endl << endl;
//            }
//                cout << "size " <<detection.size.p[2]<<endl;
//            cout << "size2 " <<detection.size.p[3]<<endl;
            for(int i =0; i < detection.size.p[2]; i++){
                float confidence = detection.at<float>(Vec<int,4>(0,0,i,2));
//                    cout << confidence << endl;
                    if(confidence > 0.6) {
                        int idx = detection.at<float>(Vec<int, 4>(0, 0, i, 1));
                        String label = classes[idx];
//                        cout << "test" << endl;
                        if (label.compare("person")) {
                            found =false;
                        continue;
                    }
                int startX = (int) (detection.at<float>(Vec<int,4>(0,0,i,3)) * frame.cols);
                int startY = (int) (detection.at<float>(Vec<int,4>(0,0,i,4)) * frame.rows);
                int endX = (int) (detection.at<float>(Vec<int,4>(0,0,i,5)) * frame.cols);
                int endY = (int) (detection.at<float>(Vec<int,4>(0,0,i,6)) * frame.rows);
//                        cout << "startX" << startX <<endl;
//                        cout << "startY" << startY <<endl;
//                        cout << "endX" << endX <<endl;
//                        cout << "endY" << endY <<endl;
                        cv::rectangle(frame, Point(startX,startY), Point(endX,endY),Scalar(0, 255, 0),2);
                        putText(frame, label, Point(startX, startY-15), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0,255,0),2);
                    }
            }






            //face recoginition 구현 중
            if(found) {
                matrix<rgb_pixel> img;
                dlib::assign_image(img, dlib::cv_image<rgb_pixel>(frame));

                dlib::array<matrix<rgb_pixel>> faces2;
                auto locations = detector(img);
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
                    cout << name <<endl;
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








//            dlib::array<matrix<rgb_pixel>> face_chips1;
//            std::vector<matrix<rgb_pixel>> face_locations1;
//            auto locations = detector(frame,1);
//            cout<< typeid(locations).name()<< endl;
//            std::vector<matrix<rgb_pixel>>  raw_landmarks;
//            for (auto face : locations) {
//                auto shape = pose_model(frame, face);
//                cout<< typeid(shape).name()<< endl;
////                raw_landmarks.push_back(shape);
//            }
//            for (auto raw : raw_landmarks)
//                matrix<rgb_pixel> face_chip;
//                net.compute_face_descriptor(frame, )
//                extract_image_chip(frame, get_face_chip_details(shape,150,0.25), face_chip);
//                face_locations1.push_back(move(face_chip));
                // Also put some boxes on the faces so we can see that the detector is finding
                // them.






//            std::vector<matrix<float,0,1>> face_descriptors1 = net(face_locations1);
//
//            std::vector<String> f_names;
//
//            for(auto f : face_descriptors1){
//                auto distance = face_encoding - f;
//                auto min_value = min(distance);
//                String name = "Unknown";
//                if(min_value < 0.6){
//                    name = "User";
//                }
//                f_names.push_back(name);
//            }
//
//            for(int i =0; i < locations.size(); i++){
//                cv::rectangle(frame, Point(locations[i].left(),locations[i].top()), Point(locations[i].right(),locations[i].bottom()),Scalar(0, 255, 0),2);
//                putText(frame, f_names[i], Point(locations[i].left() +6, locations[i].top()-6), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255,255,255),2);
//
//            }













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
// Get the names of the output layers
//std::vector<String> getOutputsNames(const Net& net)
//{
//    static std::vector<String> names;
//    if (names.empty())
//    {
//        //Get the indices of the output layers, i.e. the layers with unconnected outputs
//        std::vector<int> outLayers = net.getUnconnectedOutLayers();
//
//        //get the names of all the layers in the network
//        std::vector<String> layersNames = net.getLayerNames();
//
//        // Get the names of the output layers in names
//        names.resize(outLayers.size());
//        for (size_t i = 0; i < outLayers.size(); ++i)
//            names[i] = layersNames[outLayers[i] - 1];
//    }
//    return names;
//}




