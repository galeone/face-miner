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
    static void print(QString _testPositive, QString _testNegative, IClassifier *classifier);
};

#endif // STATS_H
