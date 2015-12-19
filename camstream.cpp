#include "camstream.h"

CamStream::CamStream(const cv::VideoCapture &cam)
{
    _cam = cam;
}

void CamStream::start()
{
    std::chrono::milliseconds time(1500); // Persistence Of Vision ~ 1/25
    while (true) {
        cv::Mat frame;
        _cam >> frame;
        emit newFrame(frame);
        std::this_thread::sleep_for(time);
    }

    emit finished();
}
