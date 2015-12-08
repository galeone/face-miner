#ifndef FEATURECLASSIFIER_H
#define FEATURECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <QString>
#include <QDirIterator>

class FeatureClassifier : IClassifier {
public:
    FeatureClassifier(std::vector<cv::Point> &, std::vector<cv::Point> &);
    void train(bool positive, QString trainingSet);
    void setConstants(cv::Mat1b &raw, uint32_t *_c1, uint32_t *_c2, uint32_t *_c3, uint32_t *_c4);
    bool classify(cv::Mat1b &window);

private:
    uint32_t _t1, _t2;
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
};

#endif // FEATURECLASSIFIER_H
