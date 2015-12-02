#ifndef CANTOR_H
#define CANTOR_H

#include <cstdint>
#include <opencv2/core.hpp>
#include <cmath>

// https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
class Cantor
{
public:
    static uint32_t pair(const cv::Point&);
    static cv::Point unpair(uint32_t z);
};

#endif // CANTOR_H
