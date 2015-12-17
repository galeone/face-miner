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
    void setConstants(cv::Mat1b &raw, double *_c1, double *_c2, double *_c3, double *_c4);
    bool classify(cv::Mat1b &window);

private:
    double _t1, _t2, _tLower[4], _tUpper[4];
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
};

#endif // FEATURECLASSIFIER_H
