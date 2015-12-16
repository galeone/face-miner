#include "stats.h"

void Stats::print(QString _testPositive, QString _testNegative, IClassifier *classifier) {
    auto truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
    QDirIterator *it = new QDirIterator(_testPositive);
    // test
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);

        if(classifier->classify(face)) {
            ++truePositive;
        } else {
            ++falseNegative;
        }
    }

    it = new QDirIterator(_testNegative);
    // test
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);

        if(classifier->classify(face)) {
            ++falsePositive;
        } else {
            ++trueNegative;
        }
    }

    std::cout << "True positive: " << truePositive << "\nTrue negative: " << trueNegative << "\nFalse positive: " << falsePositive << "\nFalse negatve: " << falseNegative <<std::endl;
    std::cout << "Precision: " << ((float)truePositive / (truePositive + falsePositive)) << std::endl;
    std::cout << "Recall: " << ((float)truePositive / (truePositive + falseNegative)) << std::endl;

    delete it;
}
