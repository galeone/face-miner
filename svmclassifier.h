#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <QString>
#include <QDirIterator>
#include "preprocessor.h"


class SVMClassifier
{
public:
    SVMClassifier(const cv::Range &rows1, const cv::Range rows2);
    bool classify(cv::Mat1b &window);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);

private:
    cv::Range _r1, _r2;
    cv::SVM *_svm;

    void _getHaarCoefficients(cv::Mat1b &window, cv::Mat1f &coeff);
};

#endif // SVMCLASSIFIER_H
