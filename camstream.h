#ifndef CamStream_H
#define CamStream_H

#include <QObject>
#include <QLabel>
#include <QString>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <thread>
#include "camstreamview.h"


class CamStream : public QObject
{
    Q_OBJECT
public:
    CamStream(const cv::VideoCapture& cam);

signals:
    void newFrame(const cv::Mat& frame);
    void finished();
    void error(QString error);

public slots:
    void start();

private:
    cv::VideoCapture _cam;
};

#endif // CamStream_H
