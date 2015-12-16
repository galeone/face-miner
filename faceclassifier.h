#ifndef FACECLASSIFIER_H
#define FACECLASSIFIER_H

#include "iclassifier.h"
#include "varianceclassifier.h"
#include "featureclassifier.h"
#include "svmclassifier.h"
#include "preprocessor.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

class FaceClassifier
{
public:
    FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size);
    bool classify(cv::Mat &);

private:
    cv::Rect _expand(cv::Rect rect, float scaleFactor);
    void _slidingSearch(cv::Mat1b &level, float factor, std::vector<cv::Rect> &allCandidates);
    VarianceClassifier *_vc;
    FeatureClassifier *_fc;
    SVMClassifier *_sc;
    cv::Size _windowSize;
    size_t _step;
};

#endif // FACECLASSIFIER_H
