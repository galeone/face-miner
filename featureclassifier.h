#ifndef FEATURECLASSIFIER_H
#define FEATURECLASSIFIER_H

#include "iclassifier.h"
#include <vector>
#include <cstdint>

class FeatureClassifier {
public:
    FeatureClassifier(std::vector<cv::Point> &, std::vector<cv::Point> &);
    void setT1(float);
    void setT2(float);
    void setData(cv::Mat &, cv::Mat &);
    float getT1();
    float getT2();
    bool rule1(float&);
    bool rule2(float&);
    bool rule3();

private:
    float _t1, _t2;
    uint32_t _c1, _c2, _c3, _c4;
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
};

#endif // FEATURECLASSIFIER_H
