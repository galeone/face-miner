#ifndef VARIANCECLASSIFIER_H
#define VARIANCECLASSIFIER_H

#include "iclassifier.h"
#include "preprocessor.h"
#include "stats.h"
#include "integralimage.h"
#include <iostream>
#include <fstream>
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VarianceClassifier : public IClassifier {
 public:
  VarianceClassifier(const cv::Size windowSize);
  bool classify(const cv::Mat1b&);
  void train(QString positiveTrainingSet, QString negativeTrainingSet);
  void train(std::vector<cv::Mat1b>& positive,
             std::vector<cv::Mat1b>& negative);
  void _getMForABC(const cv::Mat1b& window,
                   IntegralImage& ii,
                   float* ma,
                   float* mb,
                   float* mc);

 private:
  cv::Rect _A, _B, _C, _D, _E;
  float _t, _k;
};

#endif  // VARIANCECLASSIFIER_H
