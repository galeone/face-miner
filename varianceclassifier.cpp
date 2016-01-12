#include "varianceclassifier.h"

VarianceClassifier::VarianceClassifier(const cv::Size windowSize) {
  auto cols = windowSize.width, rows = windowSize.height;

  auto aThirdRows = std::floor(rows / 3),  // 6
      aThirdCols = std::floor(cols / 3);   // 6

  // Mouth region
  _E = cv::Rect(0, rows - aThirdRows, cols, aThirdRows);

  // Nose region
  _D = cv::Rect(0, rows - 2 * aThirdRows + 1, cols, aThirdRows - 1);

  // Left eye region
  auto ac_cols = aThirdCols + 1;
  auto b_cols = cols - 2 * ac_cols;

  // Top section
  if (rows % 2 != 0) {
    ++aThirdRows;
  }
  int topHeight = std::floor(aThirdRows / 3);
  _A = cv::Rect(0, 2 * topHeight, ac_cols, topHeight);

  // Between eye region
  _B = cv::Rect(ac_cols, 2 * topHeight, b_cols, topHeight);

  // Right eye region
  _C = cv::Rect(cols - ac_cols, 2 * topHeight, ac_cols, topHeight);
}

void VarianceClassifier::_getMForABC(const cv::Mat1b& window,
                                           IntegralImage& ii,
                                           float* ma,
                                           float* mb,
                                           float* mc) {
  cv::Mat1b roi_a = window(_A), roi_b = window(_B), roi_c = window(_C);
  float mu_a = ii.calcMean(_A), mu_b = ii.calcMean(_B), mu_c = ii.calcMean(_C);
  *ma = 0, *mb = 0, *mc = 0;

  uint32_t validPx = 0;

  // *ma is the average intensity of those pixels that are
  // darker than the average intensity in region A
  cv::Point coord;
  for (auto x = 0; x < roi_a.cols; ++x) {
    for (auto y = 0; y < roi_a.rows; ++y) {
      coord.x = x;
      coord.y = y;
      auto pxBrightness = roi_a.at<uchar>(coord);
      if (pxBrightness < mu_a) {
        *ma += pxBrightness;
        ++validPx;
      }
    }
  }

  *ma /= (validPx > 0 ? validPx : 1);
  validPx = 0;

  // *mc is the average intensity of those pixels that are
  // darker than the average intensity in region C
  for (auto x = 0; x < roi_c.cols; ++x) {
    for (auto y = 0; y < roi_c.rows; ++y) {
      coord.x = x;
      coord.y = y;
      auto pxBrightness = roi_c.at<uchar>(coord);
      if (pxBrightness < mu_c) {
        *mc += pxBrightness;
        ++validPx;
      }
    }
  }
  *mc /= (validPx > 0 ? validPx : 1);
  validPx = 0;

  // *mb is the average intensity of those pixels that are
  // birghter than the average intensity in region B
  for (auto x = 0; x < roi_b.cols; ++x) {
    for (auto y = 0; y < roi_b.rows; ++y) {
      coord.x = x;
      coord.y = y;
      auto pxBrightness = roi_b.at<uchar>(coord);
      if (pxBrightness > mu_b) {
        *mb += pxBrightness;
        ++validPx;
      }
    }
  }
  *mb /= (validPx > 0 ? validPx : 1);
}

// Adjust the thresholds untile the face is marked as a valid face
// we suppose that face has the same dimension of _positiveMFI / _negativeMFI
void VarianceClassifier::train(std::vector<cv::Mat1b>& positive,
                               std::vector<cv::Mat1b>& negative) {
  std::vector<double> positiveT, negativeT;
  positiveT.reserve(positive.size());
  negativeT.reserve(negative.size());

  for (const auto& raw : positive) {
    cv::Mat1b face = Preprocessor::equalize(raw);
    IntegralImage ii(face);
    positiveT.push_back(ii.calcVariance(_D));
    positiveT.push_back(ii.calcVariance(_E));
  }

  for (const auto& raw : negative) {
    cv::Mat1b face = Preprocessor::equalize(raw);
    IntegralImage ii(face);
    negativeT.push_back(ii.calcVariance(_D));
    negativeT.push_back(ii.calcVariance(_E));
  }

  _t = equal_error_rate(positiveT, negativeT).second / 21.5;
  _k = 1.65;
  std::cout << "T: << " << _t << "\nK: " << _k << std::endl;
}

void VarianceClassifier::train(QString positiveTrainingSet,
                               QString negativeTrainingSet) {
  std::vector<cv::Mat1b> positive, negative;

  QDirIterator* it = new QDirIterator(positiveTrainingSet);
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

bool VarianceClassifier::classify(const cv::Mat1b& window) {
  cv::Mat1b face = Preprocessor::equalize(window);
  IntegralImage ii(window);

  if (ii.calcVariance(_D) < _t || ii.calcVariance(_E) < _t) {
    return false;
  }

  float ma, mb, mc;
  _getMForABC(face, ii, &ma, &mb, &mc);
  if (mb < _k * ma || mb < _k * mc) {
    return false;
  }
  return true;
}
