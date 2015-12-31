#ifndef FEATURECLASSIFIER_H
#define FEATURECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include "stats.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <QString>
#include <QDirIterator>
#include <opencv2/core.hpp>

class FeatureClassifier : public IClassifier {
public:
    FeatureClassifier(std::vector<cv::Point> &, std::vector<cv::Point> &);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);
    void train(std::vector<cv::Mat1b> &truePositive, std::vector<cv::Mat1b> &falsePositive);
    bool classify(const cv::Mat1b &window) override;
    bool classify(const cv::Mat1b &window, float *_c1, float *_c2, float *_c3, float *_c4);

private:
    float _t1, _t2, _tLower[4], _tUpper[4];
    void _setConstants(const cv::Mat1b &gray, float *_c1, float *_c2, float *_c3, float *_c4);
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
};

#endif // FEATURECLASSIFIER_H
