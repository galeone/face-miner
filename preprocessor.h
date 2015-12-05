#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Preprocessor
{
public:
    static cv::Mat1b process(cv::Mat image);
    static cv::Mat1b edge(cv::Mat image);
};

#endif // PREPROCESSOR_H
