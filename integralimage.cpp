/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "integralimage.h"

void IntegralImage::setImage(const cv::Mat1b& image) {
    image.copyTo(_image);
    cv::integral(_image, _integral, _sq_integral);
}

float IntegralImage::calcMean(const cv::Rect& r) {
  int width = _integral.cols;
  unsigned int* ii1 = (unsigned int*)_integral.data;
  int a = r.x + (r.y * (width));
  int b = (r.x + r.width) + (r.y * (width));
  int c = r.x + ((r.y + r.height) * (width));
  int d = (r.x + r.width) + (r.y + r.height) * (width);
  float mx = ii1[a] + ii1[d] - ii1[b] - ii1[c];
  mx = mx / (r.width * r.height);
  return mx;
}

float IntegralImage::calcVariance(const cv::Rect &r) {
  int width = _integral.cols;
  int a = r.x + (r.y * width);
  int b = (r.x + r.width) + (r.y * width);
  int c = r.x + ((r.y + r.height) * width);
  int d = (r.x + r.width) + (r.y + r.height) * width;
  float mx = calcMean(r);
  double* ii2 = (double*)_sq_integral.data;
  float mx2 = ii2[a] + ii2[d] - ii2[b] - ii2[c];
  mx2 = mx2 / (r.width * r.height);
  mx2 = mx2 - (mx * mx);
  return mx2;
}
