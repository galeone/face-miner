#include "varianceclassifier.h"

VarianceClassifier::VarianceClassifier(const float threshold)
{
    _t = threshold;
}

bool VarianceClassifier::test(const cv::Mat &image) {
    // approximate the regions described in the paper
    // generalzing the split for every size

}
