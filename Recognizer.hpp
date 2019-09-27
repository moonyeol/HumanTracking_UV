

using namespace cv;
using namespace cv::dnn;
using namespace dlib;
using namespace std;

class Recognizer{

private:
    shape_predictor pose_model;
    anet_type faceEncoder;
    Net odNet;
    Net fdnet;
    String classesFile;
    std::vector<string> classes;
public:
    Recognizer(String landmarkDat, String frNetModel, string odConfigFile, string odWeightFile, String fdConfigFile, String fdWeightFile, String classesFile);
    ~Recognizer();
    void readClasses();
    bool humanDetection(Mat& frame);
    std::vector<dlib::rectangle> faceDetection(Mat& frame);
    dlib::array<matrix<rgb_pixel>> faceLandMark(Mat& frame,std::vector<dlib::rectangle>& locations);
    std::vector<matrix<float,0,1>> faceEncoding(dlib::array<matrix<rgb_pixel>>& faces);
};
