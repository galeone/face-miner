#ifndef STATS_H
#define STATS_H

#include <QDirIterator>
#include <QString>
#include <iostream>
#include "iclassifier.h"
#include "preprocessor.h"

class Stats
{
public:
    Stats();
    // returns the vectors of true positives and true negatives.
    static std::pair<std::vector<cv::Mat1b>, std::vector<cv::Mat1b>> test(QString _testPositive, QString _testNegative, IClassifier *classifier);
};

#endif // STATS_H
