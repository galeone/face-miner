/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef CV2QT_H
#define CV2QT_H

#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <QLabel>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Cv2Qt {
 public:
  static QImage cvMatToQImage(const cv::Mat& inMat);
  static QPixmap cvMatToQPixmap(const cv::Mat& inMat);

 public slots:
  void _updateCamView(const cv::Mat&);
  void _handleClick(const cv::Point&);

 private:
  cv::VideoCapture _cam;
  QLabel* _VideoStreamView;
};

#endif  // CV2QT_H
