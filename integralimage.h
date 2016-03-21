/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef INTEGRALIMAGE_H
#define INTEGRALIMAGE_H

// Based on: https://github.com/pi19404/OpenVision/blob/master/ImgFeatures/integralImage.cpp
#include <opencv2/imgproc.hpp>

class IntegralImage {
 public:
  cv::Mat _integral;  // integral image
  cv::Mat _sq_integral;  // sq integral image
  cv::Mat _image;  // original image
  void setImage(const cv::Mat1b &image);

  // function to compute mean value of a patch
  float calcMean(const cv::Rect &r);

  // function to compute variance of a patch
  float calcVariance(const cv::Rect &r);
};

#endif  // INTEGRALIMAGE_H
