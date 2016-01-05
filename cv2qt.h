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
