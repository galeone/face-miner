#ifndef QT2CV_H
#define QT2CV_H

#include <opencv2/core.hpp>
#include <QPoint>

class Qt2Cv {
 public:
  static cv::Point qpointToCvPoint(const QPoint& point);
};

#endif  // QT2CV_H
