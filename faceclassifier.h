/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

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

class FaceClassifier {
 public:
  FaceClassifier(VarianceClassifier* vc,
                 FeatureClassifier* fc,
                 SVMClassifier* svmc,
                 cv::Size size);
  std::vector<cv::Rect> classify(const cv::Mat&);

 private:
  void _slidingSearch(cv::Mat1b& level,
                      float factor,
                      std::vector<cv::Rect>& allCandidates);
  VarianceClassifier* _vc;
  FeatureClassifier* _fc;
  SVMClassifier* _sc;
  cv::Size _windowSize;
  size_t _step;
  float _scaleFactor;
};

#endif  // FACECLASSIFIER_H
