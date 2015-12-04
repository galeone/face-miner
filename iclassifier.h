#ifndef ICLASSIFIER_H
#define ICLASSIFIER_H

#include <opencv2/core.hpp>
#include <string>

class ILabel {
public:
    virtual ~ILabel(){}
    std::string get();
};

class IClassifier
{
public:
    virtual ~IClassifier(){}
    virtual float classify(const cv::Mat &) = 0;
    virtual void train(ILabel &, const cv::Mat &) = 0;
    virtual float getThreshold() = 0;

};

#endif // ICLASSIFIER_H
