#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _size = size;
    _step = 2;
}

//Returns true if contains some faces. Hilight with a rectangle the face on the image.
bool FaceClassifier::classify(cv::Mat1b &image) {
    //Downsample image until its dimension are less than _size (gaussian pyramid)
    //for every downsample, use a sliding window approach to extract _size window and classify it
    bool ret = false;
    cv::string wat = "name";

    std::vector<cv::Mat> pyramid;
    cv::Mat level = cv::Mat(image); // level of the pyramid
    float ratio = 1.25;
    while(level.rows >= _size.height && level.cols >= _size.width) {
        pyramid.push_back(level);
        std::cout << "Dimension: " << level.rows << "x" << level.cols << std::endl;
        cv::resize(level,level,cv::Size(0,0), ratio - 1 , ratio - 1);
    }

    for(cv::Mat level : pyramid) {
        auto name = std::string(wat.append("asd")).c_str();
        cv::namedWindow(name);
        cv::imshow(name, level);
    }

    cv::Mat1b retImage = cv::Mat(image);

    size_t iteration = 0;
    for(auto rit = pyramid.rbegin(); rit != pyramid.rend(); ++rit, ++iteration) {
        cv::Mat1b level = rit[iteration];
        for(auto x = 0; x <= level.cols - _size.width; x+=_step) {
            bool detected = false;
            for(auto y = 0; y <= level.rows - _size.height; y+=_step) {
                cv::Mat1b roi = Preprocessor::process(level(cv::Rect(x,y,_size.width, _size.height)));
                //TODO: and other 2 classificators
                //if(_vc->classify(roi)) {
                if(_fc->classify(roi)) {
                    auto scaleFactor = iteration == 0 ? 1 : std::ceil(ratio * iteration);
                    auto rect = cv::Rect(x*scaleFactor, y*scaleFactor, _size.width *scaleFactor, _size.height * scaleFactor);
                    cv::rectangle(retImage, rect ,cv::Scalar(0,0,255));
                    ret = true;
                    std::cout << iteration << " " << x << " " << y << std::endl;
                    y+=_size.height;
                    detected = true;
                }
            }
            if(detected) {
                x+=_size.width;
            }
        }
    }


    image = retImage;

    return ret;
}
