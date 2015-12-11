#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _windowSize = size;
    _step = 2;
}

cv::Rect FaceClassifier::_expand(cv::Rect rect, float scaleFactor) {
    cv::Size deltaSize( rect.width * scaleFactor, rect.height * scaleFactor );
    cv::Point offset( deltaSize.width/2, deltaSize.height/2);
    rect += deltaSize;
    rect -= offset;
    return rect;
}

//Returns true if contains some faces. Hilight with a rectangle the face on the image.
bool FaceClassifier::classify(cv::Mat &image) {

    cv::vector<cv::Rect> allCandidates;
    float scaleFactor = 1.2;
    size_t iter_count = 0;
    cv::Mat1b gray = Preprocessor::gray(image);
    for(float factor = 1; ; factor *=scaleFactor) {
        ++iter_count;
        // Size of the image scaled up
        cv::Size winSize(std::round(_windowSize.width*factor), std::round(_windowSize.height*factor));

        // Size of the image scaled down (from bigger to smaller)
        cv::Size sz(image.cols/factor, image.rows/factor);

        // Difference between sized of the scaled image and the original detection window
        cv::Size sz1(sz.width - _windowSize.width, sz.height - _windowSize.height);

        // if the actual scaled image is smaller than the origina detection window, break
        if(sz1.width < 0 || sz1.height < 0) {
            break;
        }

        cv::Mat1b level;
        cv::resize(gray,level,sz,0,0,cv::INTER_NEAREST);

        _slidingSearch(level, factor, allCandidates);

    }

    cv::groupRectangles(allCandidates, 1, 0.4);

    for(const cv::Rect &rect : allCandidates) {
        cv::rectangle(image,rect, cv::Scalar(255,255,0));
    }

    return allCandidates.size() > 0;
}

void FaceClassifier::_slidingSearch(cv::Mat1b &level, float factor, std::vector<cv::Rect> &allCandidates) {
    cv::Size winSize(_windowSize.width*factor, _windowSize.height*factor);
    for(auto x=0; x<=level.cols - _windowSize.width; x+=_step) {
        for(auto y=0; y<=level.rows - _windowSize.height; y+=_step) {
            cv::Rect roi_rect(x, y, _windowSize.width, _windowSize.height);
            cv::Mat1b roi = level(roi_rect);
            if(_vc->classify(roi) && _fc->classify(roi)) {
            //if(_vc->classify(roi)) {
            //if(_fc->classify(roi)) {
                cv::Rect destPos(std::round(x*factor), std::round(y*factor), winSize.width, winSize.height);
                allCandidates.push_back(destPos);
            }
        }
    }
}
