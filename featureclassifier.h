/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef FEATURECLASSIFIER_H
#define FEATURECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include "stats.h"
#include <vector>
#include <iostream>
#include <cstdint>
#include <QString>
#include <QDirIterator>
#include <opencv2/core.hpp>

class FeatureClassifier : public IClassifier {
 public:
  FeatureClassifier(std::vector<cv::Point>&, std::vector<cv::Point>&);
  void train(QString positiveTrainingSet, QString negativeTrainingSet);
  void train(std::vector<cv::Mat1b>& truePositive,
             std::vector<cv::Mat1b>& falsePositive);
  bool classify(const cv::Mat1b& window) override;
  bool classify(const cv::Mat1b& window,
                int32_t* _c1,
                int32_t* _c2,
                int32_t* _c3,
                int32_t* _c4);

 private:
  int32_t _t1, _t2, _tLower[4], _tUpper[4];
  void _setConstants(const cv::Mat1b& gray,
                     int32_t* _c1,
                     int32_t* _c2,
                     int32_t* _c3,
                     int32_t* _c4);
  std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
};

#endif  // FEATURECLASSIFIER_H
