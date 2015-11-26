#ifndef FACIALRECOGNITION_H
#define FACIALRECOGNITION_H

#include <QMainWindow>
#include <QMessageBox>
#include <QThread>
#include "camstream.h"
#include "videostreamview.h"
#include "cv2qt.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Ui {
class FacialRecognition;
}

class FacialRecognition : public QMainWindow
{
    Q_OBJECT

public:
    explicit FacialRecognition(QWidget *parent = 0);
    ~FacialRecognition();

private:
    Ui::FacialRecognition *ui;
    VideoStreamView* _getVideoStreamView();

private slots:
    void _updateCamView(const cv::Mat&);
    void _handleClick(const cv::Point&);
};

#endif // FACIALRECOGNITION_H
