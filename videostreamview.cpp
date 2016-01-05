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
