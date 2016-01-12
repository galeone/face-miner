#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

// Based on: https://github.com/pi19404/OpenVision/blob/master/ImgFeatures/integralImage.cpp
#include <opencv2/imgproc.hpp>

class IntegralImage {
 public:
  cv::Mat _integral;  // integral image
  cv::Mat _sq_integral;  // sq integral image
  cv::Mat _image;  // original image
  IntegralImage(const cv::Mat1b &image);

  // function to compute mean value of a patch
  float calcMean(const cv::Rect &r);

  // function to compute variance of a patch
  float calcVariance(const cv::Rect &r);
};

#endif  // INTEGRALIMAGE_H
