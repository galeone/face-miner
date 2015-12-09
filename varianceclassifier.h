#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "iclassifier.h"
#include "preprocessor.h"
#include <iostream>
#include <QString>

class VarianceClassifier : public IClassifier
{
public:
    VarianceClassifier(const cv::Size windowSize);
    bool classify(cv::Mat1b &);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);
    cv::Scalar _getMForABC(cv::Mat &window);

private:
    double _t, _k;
    cv::Rect _A, _B, _C, _D, _E;
};

#endif // VARIANCECLASSIFIER_H
