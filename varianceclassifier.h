#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include "stats.h"
#include <iostream>
#include <fstream>
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

class VarianceClassifier : public IClassifier
{
public:
    VarianceClassifier(const cv::Size windowSize, QString test_positive, QString test_negative);
    bool classify(cv::Mat1b &);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);
    cv::Scalar _getMForABC(cv::Mat1b &window);

private:
    cv::Rect _A, _B, _C, _D, _E;
    float _t;
    cv::NormalBayesClassifier *_b;
    float _k;
    QString _testPositive, _testNegative;
};

#endif // VARIANCECLASSIFIER_H
