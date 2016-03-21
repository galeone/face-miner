/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "cv2qt.h"

// Thanks to: http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/

QImage Cv2Qt::cvMatToQImage(const cv::Mat& inMat) {
  switch (inMat.type()) {
    // 8-bit, 4 channel
    case CV_8UC4: {
      QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step,
                   QImage::Format_RGB32);

      return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3: {
      QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step,
                   QImage::Format_RGB888);

      return image.rgbSwapped();
    }

    // 8-bit, 1 channel
    case CV_8UC1: {
      static QVector<QRgb> sColorTable;

      // only create our color table once
      if (sColorTable.isEmpty()) {
        for (int i = 0; i < 256; ++i)
          sColorTable.push_back(qRgb(i, i, i));
      }

      QImage image(inMat.data, inMat.cols, inMat.rows, inMat.step,
                   QImage::Format_Indexed8);

      image.setColorTable(sColorTable);

      return image;
    }

    default:
      qWarning()
          << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:"
          << inMat.type();
      break;
  }

  return QImage();
}

QPixmap Cv2Qt::cvMatToQPixmap(const cv::Mat& inMat) {
  return QPixmap::fromImage(cvMatToQImage(inMat));
}
