/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "facefinder.h"

// slot
void FaceFinder::setClassifier(FaceClassifier* fc) {
  _fc = fc;
  emit ready();
}

// emit signal found if faces are found
std::vector<std::pair<cv::Rect, cv::Mat1b>> FaceFinder::find(const cv::Mat& frame) {
  auto faces = _fc->classify(frame);
  std::vector<std::pair<cv::Rect, cv::Mat1b>> _faces;
  if (faces.size() > 0) {
    _faces.reserve(faces.size());
    for (const auto& faceRect : faces) {
      cv::Mat1b grayFrame = Preprocessor::gray(frame);
      cv::Mat1b faceTpl = Preprocessor::equalize(grayFrame(faceRect));
      _faces.push_back(std::make_pair(faceRect, faceTpl));
    }
    emit found(_faces);
  }
  // return for syncronous call
  return _faces;
}
