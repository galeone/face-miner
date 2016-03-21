/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef FACIALRECOGNITION_H
#define FACIALRECOGNITION_H

#include <QMainWindow>
#include <QMessageBox>
#include <QThread>
#include <QMimeData>
#include <QMetaObject>
#include "camstream.h"
#include "videostreamview.h"
#include "cv2qt.h"
#include "facepatternminer.h"
#include "preprocessor.h"
#include "facefinder.h"
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
  QSize* _streamSize;
  QMetaObject::Connection _trackerConnection;
  FaceFinder *_faceFinder;
  CamStream *_frameStream;
  std::vector<std::pair<cv::Rect, cv::Mat1b>> _camFaces;
  uint32_t _frameCount;

  void _updatePositivePatternStreamView(const cv::Mat&);
  void _updateNegativePatternStreamView(const cv::Mat&);
  void _startCamStream();

 private slots:
  void _updateCamView(const cv::Mat&);
  void _updateTrainingStreamView(const cv::Mat&);
  void _handleClick(const cv::Point&);
  void _track(const cv::Mat& im);
};

#endif  // FACIALRECOGNITION_H
