/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "videostreamview.h"

void VideoStreamView::setSize(const QSize& size) {
  _size = size;
}

void VideoStreamView::keyPressEvent(QKeyEvent* ev) {
  switch (ev->key()) {
    case Qt::Key::Key_Space:
      break;
  }
}

void VideoStreamView::mousePressEvent(QMouseEvent* ev) {
  emit clicked(Qt2Cv::qpointToCvPoint(ev->pos()));
}

void VideoStreamView::setImage(const QImage& image) {
  auto scaled = image.scaled(_size, Qt::KeepAspectRatio);
  auto pixmap = QPixmap::fromImage(scaled);
  QLabel::setPixmap(pixmap);
}

cv::Mat VideoStreamView::getImage() {
    return Qt2Cv::QPixmapToCvMat(*(this->pixmap()));
}
