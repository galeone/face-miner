#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size) {
    _vc = vc;
    _fc = fc;
    _sc = svmc;
    _size = size;
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
    //Downsample image until its dimension are less than _size (gaussian pyramid)
    //for every downsample, use a sliding window approach to extract _size window and classify it
    cv::Mat1b gray_img = Preprocessor::gray(image);
    cv::Mat1b level(gray_img); // level of the pyramid
    float scaleFactor = 1.25;
    auto maxlevel = 0;
    std::vector<cv::Mat1b> refs, results;
    std::vector<cv::Size> sizes;
    std::vector<cv::Rect> faces_rect;
    refs.push_back(gray_img);
    sizes.push_back(gray_img.size());
    while(level.rows/scaleFactor >= _size.height && level.cols/scaleFactor >= _size.width) {
        ++maxlevel;
        level.rows /= scaleFactor;
        level.cols /= scaleFactor;
        cv::Mat1b lvl;
        auto size = cv::Size(level.cols, level.rows);
        cv::resize(gray_img,lvl,size);
        refs.push_back(lvl);
        sizes.push_back(size);
        std::cout << level.rows << " " << level.cols << std::endl;
    }

    sizes.pop_back();

    std::cout << "levels: "<< maxlevel << std::endl;
    std::string name = "l3l";

    cv::Mat ref;
    bool found = false;

    // Process each level
    for (int level = maxlevel; level >= 0; level--) {
        ref = refs[level];

        if (level == maxlevel) {
            // sliding window
            for(auto x=0; x<=ref.cols - _size.width; x+=_step) {
                for(auto y=0; y<=ref.rows - _size.height; y+=_step) {
                    cv::Rect roi_rect(x, y, _size.width, _size.height);
                    cv::Mat1b roi(ref(roi_rect));
                    roi = Preprocessor::equalize(roi);
                    //roi = Preprocessor::edge(roi);
                    //roi = Preprocessor::threshold(roi);

                    // Propagare?
                    //if(_vc->classify(roi)) {
                    if(_vc->classify(roi) && _fc->classify(roi)) {
                        cv::Rect destPos = _expand(roi_rect,level/scaleFactor);
                        faces_rect.push_back(destPos);
                        found = true;
                        std::cout << "found " << level << std::endl;
                    }
                }
            }

        } else {
            for(auto x=0; x<=ref.cols - _size.width; x+=_step) {
                for(auto y=0; y<=ref.rows - _size.height; y+=_step) {
                    cv::Rect roi_rect(x, y, _size.width, _size.height);
                    cv::Mat1b roi = ref(roi_rect);
                    roi = Preprocessor::equalize(roi);
                    //roi = Preprocessor::edge(roi);
                    //roi = Preprocessor::threshold(roi);
                    if(_vc->classify(roi) && _fc->classify(roi)) {
                    //if(_vc->classify(roi)) {
                        cv::Rect destPos = _expand(roi_rect,level/scaleFactor);
                        faces_rect.push_back(destPos);
                        found = true;
                        std::cout << "found " << level << std::endl;
                    }
                }
            }
        }
    }

    cv::groupRectangles(faces_rect, 2, 0.25);

    for(const cv::Rect &rect : faces_rect) {
        cv::rectangle(image,rect, cv::Scalar(255,255,0));
    }

    return found;
}
