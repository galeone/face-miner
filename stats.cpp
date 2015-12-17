#include "stats.h"

// test tests classifier and returns true and false positives
std::pair<std::vector<cv::Mat1b>, std::vector<cv::Mat1b>> Stats::test(QString _testPositive, QString _testNegative, IClassifier *classifier) {
    std::vector<cv::Mat1b> truePositiveVec, falsePositiveVec;
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
            truePositiveVec.push_back(face);
        } else {
            ++falseNegative;
        }
    }
    delete it;

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
            falsePositiveVec.push_back(face);
        } else {
            ++trueNegative;
        }
    }
    delete it;

    std::cout << "True positive: " << truePositive << "\nTrue negative: " << trueNegative << "\nFalse positive: " << falsePositive << "\nFalse negatve: " << falseNegative <<std::endl;
    std::cout << "Precision: " << ((float)truePositive / (truePositive + falsePositive)) << std::endl;
    std::cout << "Recall: " << ((float)truePositive / (truePositive + falseNegative)) << std::endl;

    return std::make_pair(truePositiveVec, falsePositiveVec);
}
/*
 * Using existing trained model
True positive: 65
True negative: 23502
False positive: 71
False negatve: 407
Precision: 0.477941
Recall: 0.137712*/
