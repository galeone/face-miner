#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include <opencv2/core.hpp>

class VarianceClassifier
{
public:
    VarianceClassifier(const float);
    bool test(const cv::Mat &);

private:
    float _t;
};

#endif // VARIANCECLASSIFIER_H
