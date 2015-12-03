#include "featureclassifier.h"

void FeatureClassifier::setData(cv::Mat &raw, cv::Mat &edge) {
    _c1 = _c2 = _c3 = _c4 = 0;
    for(const cv::Point &point : _positiveMFICoordinates) {
        //c1 is the sum of pixel intesities of the positive feature pattern
        // in the raw image
        _c1 += raw.at<uchar>(point);
        //c3 is the sum of pixel intesities of the positive faeture pattern
        // in the edge image
        _c3 += edge.at<uchar>(point);
    }

    for(const cv::Point &point : _negativeMFICoordinates) {
        //c2 is the sum of pixel intesities of the negatie feature pattern
        // in the raw image
        _c2 += raw.at<uchar>(point);
        //c4 is the sum of pixel intesities of the negative faeture pattern
        // in the edge image
        _c4 += edge.at<uchar>(point);
    }
}

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = _t2 = 0;
}

bool FeatureClassifier::rule1(float &diff) {
    diff = _c1 - _c2;
    return diff > _t1;
}

bool FeatureClassifier::rule2(float &diff) {
    diff = _c3 - _c4;
    return diff > _t2;
}

bool FeatureClassifier::rule3() {
    // TODO: rule 3
    return true;
}

float FeatureClassifier::getT1() {
    return _t1;
}

float FeatureClassifier::getT2() {
    return _t2;
}

void FeatureClassifier::setT1(float t1) {
    _t1 = t1;
}

void FeatureClassifier::setT2(float t2) {
    _t2 = t2;
}
