#include "qt2cv.h"

cv::Point Qt2Cv::qpointToCvPoint(const QPoint& point)
{
    return cv::Point(point.x(), point.y());
}
