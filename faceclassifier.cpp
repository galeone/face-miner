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
    cv::Mat1b gray_img = Preprocessor::gray(cv::Mat(image));
    cv::Mat1b level(gray_img); // level of the pyramid
    float scaleFactor = 2;
    auto maxlevel = 0;
    cv::Size smallerLayerSize;
    while(level.rows/scaleFactor >= _size.height && level.cols/scaleFactor >= _size.width) {
        ++maxlevel;
        level.rows /= scaleFactor;
        level.cols /= scaleFactor;
        std::cout << level.rows << " " << level.cols << std::endl;
        smallerLayerSize.height = level.rows;
        smallerLayerSize.width = level.cols;
    }

    std::cout << "levels: "<< maxlevel << std::endl;

    std::vector<cv::Mat1b> refs, results;

    // Build Gaussian pyramid
    cv::buildPyramid(gray_img, refs, maxlevel);

    cv::Mat ref, res;
    bool found = false;

    // Process each level
    for (int level = maxlevel; level >= 0; level--) {
        ref = refs[level];
        //tpl = tpls[level];
        std::cout << ref.size() << " < elelel" << std::endl;
        res = cv::Mat1b::zeros(ref.size());

        if (level == maxlevel) {
            // On the smallest level, just perform regular template matching
            //cv::matchTemplate(ref, tpl, res, CV_TM_CCORR_NORMED);
            // sliding window

            for(auto x=0; x<=ref.cols - _size.width; x+=_step) {
                for(auto y=0; y<=ref.rows - _size.height; y+=_step) {
                    cv::Rect roi_rect(x, y, _size.width, _size.height);
                    cv::Mat1b roi = ref(roi_rect);
                    //if(_vc->classify(roi) && _fc->classify(roi)) {
                     if(_vc->classify(roi)) {
                        roi.copyTo(res(roi_rect));
                        found = true;
                    }
                }
            }

        } else {
            // On the next layers, template matching is performed on pre-defined
            // ROI areas.  We define the ROI using the template matching result
            // from the previous layer.

            cv::Mat1b mask8u;
            cv::pyrUp(results.back(), mask8u);
            mask8u.copyTo(image);
            return true;

            // Find matches from previous layer
            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(mask8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

            // Use the contours to define region of interest and
            // perform template matching on the areas
            for (size_t i = 0; i < contours.size(); i++) {
                cv::Rect r = cv::boundingRect(contours[i]);
                /*cv::matchTemplate(
                    ref(r + (_size - cv::Size(1,1))),
                    tpl,
                    res(r),
                    CV_TM_CCORR_NORMED
                );*/
                cv::Mat1b roiBig = Preprocessor::process(ref(r));
                cv::Mat1b roiForClassifiers;
                //cv::pyrDown(roiBig,roiForClassifiers,smallerLayerSize);
                //cv::resize(roiForClassifiers,roiForClassifiers,_size);
                cv::resize(roiBig,roiForClassifiers,_size);
                // if(_vc->classify(roiForClassifiers) && _fc->classify(roiForClassifiers)) {
                if(_vc->classify(roiForClassifiers)) {
                    roiBig.copyTo(res(r));
                    found = true;
                }

                /*
                for(auto x=0; x<=ref.cols - _size.width; x+=_step) {
                    for(auto y=0; y<=ref.rows - _size.height; y+=_step) {
                        cv::Mat1b roi = Preprocessor::process(ref(r));
                        if(_vc->classify(roi) && _fc->classify(roi)) {
                            roi.copyTo(res(r));
                            found = true;
                        }
                    }
                }
                */
            }
        }

        // Only keep good matches
        cv::threshold(res, res, 0.94, 1., CV_THRESH_TOZERO);
        results.push_back(res);
    }

    while (true) {
        double minval, maxval;
        cv::Point minloc, maxloc;
        cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);

        if (maxval >= 0.9) {
            cv::rectangle(image, maxloc,
                          cv::Point(maxloc.x + _size.width, maxloc.y + _size.height),
                          CV_RGB(0,255,0), 2);
            cv::floodFill(res, maxloc,
                          cv::Scalar(0), 0,
                          cv::Scalar(.1),
                          cv::Scalar(1.));
        } else {
            break;
        }
    }
    return found;
}

