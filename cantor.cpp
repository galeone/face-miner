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
