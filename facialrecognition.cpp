#include "facialrecognition.h"
#include "ui_facialrecognition.h"

FacialRecognition::FacialRecognition(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::FacialRecognition)
{
    ui->setupUi(this);

    cv::VideoCapture _cam(0);
    if (!_cam.isOpened()) {
        QMessageBox error(this);
        error.critical(this, "Error", "Unable to open webcam");
        error.show();
    }



    // register cv::Mat type, thus it can be used in signals and slots
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QThread *frameStreamThread = new QThread();
    CamStream *frameStream = new CamStream(_cam);
    // Move the object frameStream to his own thread
    frameStream->moveToThread(frameStreamThread);

    // sending CamStream::newFrame output, into FacialRecognition::updateCamView input
    connect(frameStream, &CamStream::newFrame, this, &FacialRecognition::_updateCamView);

    // start frameStream on frameStreamThread start
    connect(frameStreamThread, &QThread::started, frameStream, &CamStream::start);


    // sending click coordinates into CamStremView to FacialRecognition::_handleClick
    connect(_getVideoStreamView(),&VideoStreamView::clicked, this, &FacialRecognition::_handleClick);

    // start thread: stream of frames
    frameStreamThread->start();
}

void FacialRecognition::_updateCamView(const cv::Mat& frame)
{
    QImage camImage = Cv2Qt::cvMatToQImage(frame);
    _getVideoStreamView()->setPixmap(QPixmap::fromImage(camImage));
}

void FacialRecognition::_handleClick(const cv::Point& point)
{
    std::cout << point << std::endl;
}

FacialRecognition::~FacialRecognition()
{
    delete ui;
}

VideoStreamView* FacialRecognition::_getVideoStreamView() {
    return reinterpret_cast<VideoStreamView*>(ui->gridLayout->itemAtPosition(0,0)->widget());
}
