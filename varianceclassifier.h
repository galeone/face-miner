#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include "stats.h"
#include <iostream>
#include <fstream>
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VarianceClassifier : public IClassifier
{
public:
    VarianceClassifier(const cv::Size windowSize);
    bool classify(const cv::Mat1b &);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);
    void train(std::vector<cv::Mat1b> &positive,std::vector<cv::Mat1b> &negative);
    cv::Scalar _getMForABC(const cv::Mat1b &window);

private:
    cv::Rect _A, _B, _C, _D, _E;
    float _t;
    float _k;
};

#endif // VARIANCECLASSIFIER_H
