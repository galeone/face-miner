#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _size = size;
    _step = 2;
}

//Returns true if contains some faces. Hilight with a rectangle the face on the image.
bool FaceClassifier::classify(cv::Mat &image) {
    //Downsample image until its dimension are less than _size (gaussian pyramid)
    //for every downsample, use a sliding window approach to extract _size window and classify it
    bool ret = false;

    std::vector<cv::Mat> pyramid;
    cv::Mat level = cv::Mat(image); // level of the pyramid
    float ratio = 1.25;
    while(level.rows >= _size.height && level.cols >= _size.width) {
        pyramid.push_back(level);
        std::cout << "Dimension: " << level.rows << "x" << level.cols << std::endl;
        cv::resize(level,level,cv::Size(0,0), ratio - 1 , ratio - 1);
    }

    cv::Mat retImage = image.clone();

    size_t iteration = 0;
    // from the smallest to the highest
    for(auto it = pyramid.begin(); it != pyramid.end(); ++it, ++iteration) {
        cv::Mat level = *it;
        for(auto x = 0; x <= level.cols - _size.width; x+=_step) {
            bool detected = false;
            for(auto y = 0; y <= level.rows - _size.height; y+=_step) {
                //cv::Mat1b roi = Preprocessor::process(level(cv::Rect(x,y,_size.width, _size.height)));
                cv::Mat1b roi = Preprocessor::gray(level(cv::Rect(x,y,_size.width, _size.height)));
                //TODO: and other 2 classificators
                if(_fc->classify(roi) /*&& _vc->classify(roi)*/) {
                    auto scaleFactor = std::ceil(ratio * pyramid.size() - iteration + 1);
                    auto rect = cv::Rect(x*scaleFactor, y*scaleFactor, _size.width *scaleFactor, _size.height * scaleFactor);
                    cv::rectangle(retImage, rect ,cv::Scalar(0,255,255));
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
