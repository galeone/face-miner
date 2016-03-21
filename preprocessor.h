/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

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
