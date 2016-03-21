/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef VideoStreamView_H
#define VideoStreamView_H

#include <QLabel>
#include <Qt>
#include <QKeyEvent>
#include <QMouseEvent>
#include "qt2cv.h"
#include <iostream>

class VideoStreamView : public QLabel {
  Q_OBJECT

 public:
  explicit VideoStreamView(QWidget* parent = 0) : QLabel(parent) {}
  void setSize(const QSize& size);
  void setImage(const QImage& image);
  cv::Mat getImage();

 signals:
  void clicked(const cv::Point& position);

 protected:
  virtual void keyPressEvent(QKeyEvent* ev);
  virtual void mousePressEvent(QMouseEvent* ev);

 private:
  QSize _size;
};

#endif  // VideoStreamView_H
