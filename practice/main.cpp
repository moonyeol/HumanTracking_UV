#include <iostream>
#include <algorithm>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <opencv4/opencv2/highgui/highgui.hpp>

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

std::tuple<long, long, long, long> _rect_to_css(dlib::rectangle rect){
    return std::make_tuple(rect.left(),rect.top(),rect.right(),rect.bottom());
}

dlib::rectangle _css_to_rect(long left, long top, long right, long bottom){
    return dlib::rectangle(left,top,right,bottom);
}

std::tuple<long, long, long, long> _trim_css_to_bounds(dlib::rectangle rect,cv::Mat img){
    return std::make_tuple(max(rect.left(),(long)0), max(rect.top(),(long)0), min(rect.right(), img.get),min(rect.right(),img[0]));
}

std::vector<matrix<rgb_pixel>> jitter_image(
        const matrix<rgb_pixel>& img
);



int main() try{
    std::cout << "Hello, World!" << std::endl;

    frontal_face_detector detector = get_frontal_face_detector();

    shape_predictor sp;
    deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;

    anet_type net;
    deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
    load_image(img, "pic/user.jpg");
    image_window win(img);

    std::vector<matrix<rgb_pixel>> faces;

    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.
        win.add_overlay(face);
    }

    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }

    std::vector<matrix<float,0,1>> face_descriptors = net(faces);

    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    cout << "number of people found in the image: "<< num_clusters << endl;

    std::vector<image_window> win_clusters(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels.size(); ++j)
        {
            if (cluster_id == labels[j])
                temp.push_back(faces[j]);
        }
        win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters[cluster_id].set_image(tile_images(temp));
    }
    cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;
    matrix<float,0,1> face_descriptor = mean(mat(net(jitter_image(faces[0]))));
    cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    cout << "hit enter to terminate" << endl;
    cin.get();

    return 0;
}catch (std::exception& e)
{
    cout << e.what() << endl;
}
std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img){
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}



