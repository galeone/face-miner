#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(
    std::vector<cv::Point>& positiveMFICoordinates,
    std::vector<cv::Point>& negativeMFICoordinates) {
  _positiveMFICoordinates = positiveMFICoordinates;
  _negativeMFICoordinates = negativeMFICoordinates;
  _t1 = 0;
  _t2 = 0;

  for (auto i = 0; i < 4; ++i) {
    _tLower[i] = 0;
    _tUpper[i] = 0;
  }
}

void FeatureClassifier::_setConstants(const cv::Mat1b& gray,
                                      int32_t* _c1,
                                      int32_t* _c2,
                                      int32_t* _c3,
                                      int32_t* _c4) {
  cv::Mat1b edge = Preprocessor::equalize(gray);
  edge = Preprocessor::edge(edge);

  *_c1 = *_c2 = *_c3 = *_c4 = 0;
  for (const cv::Point& point : _positiveMFICoordinates) {
    // c1 is the sum of pixel intesities of the positive feature pattern
    // in the gray image
    *_c1 += gray.at<uchar>(point);
    // c3 is the sum of pixel intesities of the positive faeture pattern
    // in the edge image
    *_c3 += edge.at<uchar>(point);
  }

  for (const cv::Point& point : _negativeMFICoordinates) {
    // c2 is the sum of pixel intesities of the negative feature pattern
    // in the gray image
    *_c2 += gray.at<uchar>(point);
    // c4 is the sum of pixel intesities of the negative faeture pattern
    // in the edge image
    *_c4 += edge.at<uchar>(point);
  }
}

void FeatureClassifier::train(std::vector<cv::Mat1b>& truePositive,
                              std::vector<cv::Mat1b>& falsePositive) {
  std::vector<int32_t> positiveT1, positiveT2, positiveCoeff[4];
  int32_t _c1, _c2, _c3, _c4;

  for (const auto& gray : truePositive) {
    _setConstants(gray, &_c1, &_c2, &_c3, &_c4);

    positiveT1.push_back(_c1 - _c2);
    positiveT2.push_back(_c3 - _c4);

    positiveCoeff[0].push_back(_c1);
    positiveCoeff[1].push_back(_c2);
    positiveCoeff[2].push_back(_c3);
    positiveCoeff[3].push_back(_c4);
  }

  for (auto i = 0; i < 4; ++i) {
    std::sort(positiveCoeff[i].begin(), positiveCoeff[i].end());
    positiveCoeff[i].erase(
        std::unique(positiveCoeff[i].begin(), positiveCoeff[i].end()),
        positiveCoeff[i].end());
    size_t size = positiveCoeff[i].size();
    size_t elm = size / 9;
    std::cout << "No duplicates: " << size << "\n";
    _tLower[i] = std::accumulate(positiveCoeff[i].begin() + 1,
                                 positiveCoeff[i].begin() + elm + 1, 0.0f) /
                 elm;
    _tUpper[i] = std::accumulate(positiveCoeff[i].end() - elm - 1,
                                 positiveCoeff[i].end() - 1, 0.0f) /
                 elm;
  }

  std::sort(positiveT1.begin(), positiveT1.end());
  positiveT1.erase(std::unique(positiveT1.begin(), positiveT1.end()),
                   positiveT1.end());
  size_t size = positiveT1.size();
  std::cout << "Positive T1 size: " << size << "\n";
  size_t elm = size / 9;
  _t1 = std::accumulate(positiveT1.begin() + 1, positiveT1.begin() + elm + 1,
                        0.0f) /
        elm;

  std::sort(positiveT2.begin(), positiveT2.end());
  positiveT2.erase(std::unique(positiveT2.begin() + 1, positiveT2.end() + 1),
                   positiveT2.end());
  size = positiveT2.size();
  std::cout << "Positive T2 size: " << size << "\n";
  elm = size / 9;
  _t2 = std::accumulate(positiveT2.begin() + 1, positiveT2.begin() + elm + 1,
                        0.0f) /
        elm;

  _tUpper[1] += 100;
  _tLower[1] -= 100;

//  _tUpper[2] -= 100;

  _tLower[3] -= 100;
  _tUpper[3] += 100;
  _t2 -= 100;
  _t1 -= 50;

  std::cout << "T1: " << _t1 << "\nT2: " << _t2 << "\n";
  for (auto i = 0; i < 4; ++i) {
    std::cout << "T_lower{" << i << "} = " << _tLower[i] << "\n";
    std::cout << "T_upper{" << i << "} = " << _tUpper[i] << "\n";
  }
  std::cout << std::endl;
}

void FeatureClassifier::train(QString positiveTrainingSet,
                              QString negativeTrainingSet) {
  QDirIterator* it = new QDirIterator(positiveTrainingSet);
  std::vector<cv::Mat1b> positive, negative;
  while (it->hasNext()) {
    auto fileName = it->next();
    if (!Preprocessor::validMime(fileName)) {
      continue;
    }
    cv::Mat raw = cv::imread(fileName.toStdString());
    cv::Mat1b gray = Preprocessor::gray(raw);
    positive.push_back(gray);
  }

  delete it;

  it = new QDirIterator(negativeTrainingSet);
  while (it->hasNext()) {
    auto fileName = it->next();
    if (!Preprocessor::validMime(fileName)) {
      continue;
    }

    cv::Mat raw = cv::imread(fileName.toStdString());
    cv::Mat1b gray = Preprocessor::gray(raw);
    negative.push_back(gray);
  }

  delete it;

  return train(positive, negative);
}

// Classify suppose a gray window
bool FeatureClassifier::classify(const cv::Mat1b& window,
                                 int32_t* _c1,
                                 int32_t* _c2,
                                 int32_t* _c3,
                                 int32_t* _c4) {
  _setConstants(window, _c1, _c2, _c3, _c4);

  return *_c1 - *_c2 >= _t1 && *_c3 - *_c4 >= _t2 && _tLower[0] <= *_c1 &&
         _tUpper[0] >= *_c1 && _tLower[1] <= *_c2 && _tUpper[1] >= *_c2 &&
         _tLower[2] <= *_c3 && _tUpper[2] >= *_c3 && _tLower[3] <= *_c4 &&
         _tUpper[3] >= *_c4;
}
bool FeatureClassifier::classify(const cv::Mat1b& window) {
  int32_t a, b, c, d;
  return classify(window, &a, &b, &c, &d);
}
