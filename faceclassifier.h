#ifndef FACECLASSIFIER_H
#define FACECLASSIFIER_H

#include "iclassifier.h"
#include "varianceclassifier.h"
#include "featureclassifier.h"
#include "svmclassifier.h"
#include "preprocessor.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class FaceClassifier : public IClassifier
{
public:
    FaceClassifier(VarianceClassifier *vc, FeatureClassifier *fc, SVMClassifier *svmc, cv::Size size);
    bool classify(cv::Mat1b &) override;

private:
    VarianceClassifier *_vc;
    FeatureClassifier *_fc;
    SVMClassifier *_sc;
    cv::Size _size;
    size_t _step;
};

#endif // FACECLASSIFIER_H
