/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef SVMCLASSIFIER_H
#define SVMCLASSIFIER_H

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <QString>
#include <QDirIterator>
#include "preprocessor.h"
#include "stats.h"
#include "iclassifier.h"

class SVMClassifier : public IClassifier {
 public:
  SVMClassifier(const cv::Rect& rows1, const cv::Rect& rows2);
  bool classify(const cv::Mat1b& window);
  void train(std::vector<cv::Mat1b>& truePositive,
             std::vector<cv::Mat1b>& falsePositive);
  void train(QString positiveTrainingSet, QString negativeTrainingSet);

 private:
  cv::Rect _r1, _r2;
  cv::Ptr<cv::ml::SVM> _svm;
  cv::PCA* _pca;
  uint32_t _featureVectorCard;
  int _egVectorCard;

  void _getFeatures(const cv::Mat1b& window, cv::Mat1f& coeff);
  void _insertLineAtPosition(const cv::Mat1f& source,
                             cv::Mat1f& dest,
                             uint32_t position);
  static void _haar_2d(int m, int n, double u[]);
};

#endif  // SVMCLASSIFIER_H
