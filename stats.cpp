#include "stats.h"

// test tests classifier and returns true and false positives
std::pair<std::vector<cv::Mat1b>, std::vector<cv::Mat1b>> Stats::test(
    QString _testPositive,
    QString _testNegative,
    IClassifier* classifier) {
  std::vector<cv::Mat1b> truePositiveVec, falsePositiveVec;
  auto truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
  QDirIterator* it = new QDirIterator(_testPositive);
  // test
  while (it->hasNext()) {
    auto fileName = it->next();
    if (!Preprocessor::validMime(fileName)) {
      continue;
    }

    cv::Mat raw = cv::imread(fileName.toStdString());
    cv::Mat1b gray = Preprocessor::gray(raw);

    if (classifier->classify(gray)) {
      ++truePositive;
      truePositiveVec.push_back(gray);
    } else {
      ++falseNegative;
    }
  }
  delete it;

  it = new QDirIterator(_testNegative);
  // test
  while (it->hasNext()) {
    auto fileName = it->next();
    if (!Preprocessor::validMime(fileName)) {
      continue;
    }

    cv::Mat raw = cv::imread(fileName.toStdString());
    cv::Mat1b gray = Preprocessor::gray(raw);

    if (classifier->classify(gray)) {
      ++falsePositive;
      falsePositiveVec.push_back(gray);
    } else {
      ++trueNegative;
    }
  }
  delete it;

  std::cout << "True positive: " << truePositive
            << "\nTrue negative: " << trueNegative
            << "\nFalse positive: " << falsePositive
            << "\nFalse negatve: " << falseNegative << "\n";
  std::cout << "Precision: "
            << ((float)truePositive / (truePositive + falsePositive)) << "\n";
  std::cout << "Recall (=positive detection rate): "
            << ((float)truePositive / (truePositive + falseNegative)) << "\n";
  std::cout << "Negative detection rate: "
            << ((float)trueNegative / (trueNegative + falsePositive))
            << std::endl;

  return std::make_pair(truePositiveVec, falsePositiveVec);
}
