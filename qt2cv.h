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
