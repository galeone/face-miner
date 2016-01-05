#ifndef FACIALRECOGNITION_H
#define FACIALRECOGNITION_H

#include <QMainWindow>
#include <QMessageBox>
#include <QThread>
#include <QMimeData>
#include "camstream.h"
#include "videostreamview.h"
#include "cv2qt.h"
#include "facepatternminer.h"
#include "preprocessor.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Ui {
class FacialRecognition;
}

class FacialRecognition : public QMainWindow {
  Q_OBJECT

 public:
  explicit FacialRecognition(QWidget* parent = 0);
  ~FacialRecognition();

 private:
  Ui::FacialRecognition* ui;
  VideoStreamView* _getCamStreamView();
  VideoStreamView* _getTrainingStreamView();
  VideoStreamView* _getPositivePatternStreamView();
  VideoStreamView* _getNegativePatternStreamView();
  FaceClassifier* _faceClassifier;
  std::vector<std::pair<cv::Rect, cv::Mat1b>> _camFaces;
  QSize* _streamSize;

  void _updatePositivePatternStreamView(const cv::Mat&);
  void _updateNegativePatternStreamView(const cv::Mat&);
  void _track(const cv::Mat& im);
  void _startCamStream();

 private slots:
  void _updateCamView(const cv::Mat&);
  void _updateTrainingStreamView(const cv::Mat&);
  void _handleClick(const cv::Point&);
};

#endif  // FACIALRECOGNITION_H
