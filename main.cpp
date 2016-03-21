/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "facialrecognition.h"
#include <QApplication>
#include <opencv2/core/ocl.hpp>

int main(int argc, char* argv[]) {
  // Enable OpenCL acceleration
  cv::ocl::setUseOpenCL(true);
  QApplication a(argc, argv);
  FacialRecognition w;
  w.show();

  return a.exec();
}
