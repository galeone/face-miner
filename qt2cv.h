/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef QT2CV_H
#define QT2CV_H

#include <QImage>
#include <QPixmap>
#include <QPoint>
#include <QDebug>
#include <QLabel>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
class Qt2Cv {
 public:
  static cv::Point qpointToCvPoint(const QPoint& point);
  static cv::Mat QImageToCvMat(const QImage& inImage,
                               bool inCloneImageData = true);
  static cv::Mat QPixmapToCvMat(const QPixmap& inPixmap,
                                bool inCloneImageData = true);
};

#endif  // QT2CV_H
