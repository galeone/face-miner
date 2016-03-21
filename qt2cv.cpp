/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

// Thanks to: http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/

#include "qt2cv.h"

cv::Point Qt2Cv::qpointToCvPoint(const QPoint& point) {
  return cv::Point(point.x(), point.y());
}


cv::Mat Qt2Cv::QImageToCvMat(const QImage& inImage, bool inCloneImageData) {
  switch (inImage.format()) {
    // 8-bit, 4 channel
    case QImage::Format_RGB32: {
      cv::Mat mat(inImage.height(), inImage.width(), CV_8UC4,
                  const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine());

      return (inCloneImageData ? mat.clone() : mat);
    }

    // 8-bit, 3 channel
    case QImage::Format_RGB888: {
      if (!inCloneImageData)
        qWarning() << "ASM::QImageToCvMat() - Conversion requires cloning "
                      "since we use a temporary QImage";

      QImage swapped = inImage.rgbSwapped();

      return cv::Mat(swapped.height(), swapped.width(), CV_8UC3,
                     const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine())
          .clone();
    }

    // 8-bit, 1 channel
    case QImage::Format_Indexed8: {
      cv::Mat mat(inImage.height(), inImage.width(), CV_8UC1,
                  const_cast<uchar*>(inImage.bits()), inImage.bytesPerLine());

      return (inCloneImageData ? mat.clone() : mat);
    }

    default:
      qWarning()
          << "ASM::QImageToCvMat() - QImage format not handled in switch:"
          << inImage.format();
      break;
  }

  return cv::Mat();
}

// If inPixmap exists for the lifetime of the resulting cv::Mat, pass false to
// inCloneImageData to share inPixmap's data
// with the cv::Mat directly
//    NOTE: Format_RGB888 is an exception since we need to use a local QImage
//    and thus must clone the data regardless
cv::Mat Qt2Cv::QPixmapToCvMat(const QPixmap& inPixmap, bool inCloneImageData) {
  return QImageToCvMat(inPixmap.toImage(), inCloneImageData);
}
