/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef ICLASSIFIER_H
#define ICLASSIFIER_H

#include <opencv2/core.hpp>

class IClassifier {
 public:
  virtual ~IClassifier() {}
  virtual bool classify(const cv::Mat1b&) = 0;
};

#endif  // ICLASSIFIER_H
