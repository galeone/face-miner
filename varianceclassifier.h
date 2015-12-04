#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "iclassifier.h"
#include <iostream>

class VarianceClassifier : public IClassifier
{
public:
    VarianceClassifier(const cv::Mat &, const cv::Mat &);
    bool classify(cv::Mat &) override;
    void train(cv::Mat1b &face);
    cv::Scalar _getMForABC(cv::Mat &window);

private:
    cv::Mat _positiveMFI, _negativeMFI;
    float _t, _k;
    uint32_t _trainingNumber;
    cv::Rect _A, _B, _C, _D, _E;
};

#endif // VARIANCECLASSIFIER_H
