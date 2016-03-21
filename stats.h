/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef STATS_H
#define STATS_H

#include <QDirIterator>
#include <QString>
#include <iostream>
#include "iclassifier.h"
#include "preprocessor.h"

class Stats {
 public:
  Stats();
  // returns the vectors of true positives and true negatives.
  static std::pair<std::vector<cv::Mat1b>, std::vector<cv::Mat1b>>
  test(QString _testPositive, QString _testNegative, IClassifier* classifier);
};

#endif  // STATS_H
