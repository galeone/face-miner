/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef CANTOR_H
#define CANTOR_H

#include <cstdint>
#include <opencv2/core.hpp>
#include <cmath>

// https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
class Cantor {
 public:
  static uint32_t pair(const cv::Point&);
  static cv::Point unpair(uint32_t z);
};

#endif  // CANTOR_H
