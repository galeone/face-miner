/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef CamStream_H
#define CamStream_H

#include <QObject>
#include <QLabel>
#include <QString>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <thread>
#include "videostreamview.h"

class CamStream : public QObject {
  Q_OBJECT
 public:
  CamStream(const cv::VideoCapture& cam);

 signals:
  void newFrame(const cv::Mat& frame);
  void finished();
  void error(QString error);

 public slots:
  void start();

 private:
  cv::VideoCapture _cam;
};

#endif  // CamStream_H
