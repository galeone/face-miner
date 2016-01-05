#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QString>
#include <QDirIterator>
#include <QMimeDatabase>

class Preprocessor {
 public:
  static cv::Mat1b process(const cv::Mat& image);
  static cv::Mat1b edge(const cv::Mat& image);
  static cv::Mat1b gray(const cv::Mat& image);
  static cv::Mat1b equalize(const cv::Mat1b& gray);
  static cv::Mat1b threshold(const cv::Mat1b& grad);
  static bool validMime(QString fileName,
                        QString _mimeFilter = "image/x-portable-graymap");
};

#endif  // PREPROCESSOR_H
