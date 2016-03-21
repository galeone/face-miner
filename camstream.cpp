/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "camstream.h"

CamStream::CamStream(const cv::VideoCapture& cam) {
  _cam = cam;
}

void CamStream::start() {
  std::chrono::milliseconds time(40);  // Persistence Of Vision ~ 1/25
  while (true) {
    cv::Mat frame;
    _cam >> frame;
    emit newFrame(frame);
    std::this_thread::sleep_for(time);
  }

  emit finished();
}
