#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <QString>
#include <QDirIterator>
#include "preprocessor.h"
#include "stats.h"
#include "iclassifier.h"


class SVMClassifier : public IClassifier
{
public:
    SVMClassifier(const cv::Rect &rows1, const cv::Rect &rows2);
    bool classify(const cv::Mat1b &window);
    void train(std::vector<cv::Mat1b> &truePositive, std::vector<cv::Mat1b> &falsePositive);
    void train(QString positiveTrainingSet, QString negativeTrainingSet);

private:
    cv::Rect _r1, _r2;
    cv::Ptr<cv::ml::SVM> _svm;
    uint32_t _featureVectorCard;

    void _getFeatures(const cv::Mat1b &window, cv::Mat1f &coeff);
    void _insertLineAtPosition(const cv::Mat1f &source, cv::Mat1f &dest, uint32_t position);
    static void _haarWavelet(cv::Mat src, cv::Mat &dst, int NIter);
};

#endif // SVMCLASSIFIER_H
