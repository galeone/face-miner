/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#include "cantor.h"

uint32_t Cantor::pair(const cv::Point& point) {
  auto k1 = point.x, k2 = point.y;
  auto sum = k1 + k2;
  return (sum * (sum + 1)) / 2 + k2;
}

cv::Point Cantor::unpair(uint32_t z) {
  auto w = std::floor((std::sqrt(8 * z + 1) - 1) / 2);
  auto t = std::floor((w * w + w) / 2);
  auto y = z - t;
  auto x = w - y;
  return cv::Point(x, y);
}
