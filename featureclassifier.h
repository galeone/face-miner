#ifndef FEATURECLASSIFIER_H
#define FEATURECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <QString>
#include <QDirIterator>
#include <opencv2/ml.hpp>

class FeatureClassifier : IClassifier {
public:
    FeatureClassifier(std::vector<cv::Point> &, std::vector<cv::Point> &);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);
    void setConstants(cv::Mat1b &raw, int32_t *_c1, int32_t *_c2, int32_t *_c3, int32_t *_c4);
    bool classify(cv::Mat1b &window);

private:
    cv::Boost *_t1, *_t2, *_tLower[4], *_tUpper[4];
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
    float _delta;
};

#endif // FEATURECLASSIFIER_H
