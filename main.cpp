#include "facialrecognition.h"
#include <QApplication>
#include <opencv2/core/ocl.hpp>

int main(int argc, char *argv[])
{
    // Enable OpenCL acceleration
    cv::ocl::setUseOpenCL(true);
    QApplication a(argc, argv);
    FacialRecognition w;
    w.show();

    return a.exec();
}
