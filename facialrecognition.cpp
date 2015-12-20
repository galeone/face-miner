#include "facialrecognition.h"
#include "ui_facialrecognition.h"

FacialRecognition::FacialRecognition(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::FacialRecognition)
{
    ui->setupUi(this);
    this->showMaximized();

    // register cv::Mat type, thus it can be used in signals and slots
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QSize streamSize(300,300);
/*
    cv::VideoCapture _cam(0);

    if (!_cam.isOpened()) {
        QMessageBox error(this);
        error.critical(this, "Error", "Unable to open webcam");
        error.show();
    } else {
        QThread *frameStreamThread = new QThread();
        CamStream *frameStream = new CamStream(_cam);

        // Move the object frameStream to his own thread
        frameStream->moveToThread(frameStreamThread);

        // sending CamStream::newFrame output, into FacialRecognition::updateCamView input
        connect(frameStream, &CamStream::newFrame, this, &FacialRecognition::_updateCamView);

        // start frameStream on frameStreamThread start
        connect(frameStreamThread, &QThread::started, frameStream, &CamStream::start);

        // sending click coordinates into CamStremView to FacialRecognition::_handleClick
        connect(_getCamStreamView(),&VideoStreamView::clicked, this, &FacialRecognition::_handleClick);

        // set CamStreamView to fixed size
        _getCamStreamView()->setSize(streamSize);

        // start thread: stream of frames
        frameStreamThread->start();
    }
*/
    // set TraingStreamView to fixed size
    _getTrainingStreamView()->setSize(streamSize);

    _getPositivePatternStreamView()->setSize(streamSize);
    _getNegativePatternStreamView()->setSize(streamSize);

    // Create a thread for the miner
    QThread *minerThread = new QThread();

    // Create the face pattern miner
    // TODO: make the dataset with positive and negative items selectable from the view
    FacePatternMiner *patternMiner = new FacePatternMiner(
                "./datasets/mitcbcl/train/face/",
                "./datasets/mitcbcl/train/non-face/",
                "./datasets/mitcbcl/test/face/",
                "./datasets/mitcbcl/test/non-face/");

    // move the miner to his own thread
    patternMiner->moveToThread(minerThread);

    // connect signal start of the thread to the start() method of the miner
    connect(minerThread, &QThread::started, patternMiner, &FacePatternMiner::start);

    // connect preprocessing signal of miner to the GUI, to show what image has being processed
    connect(patternMiner, &FacePatternMiner::preprocessing, this, &FacialRecognition::_updateTrainingStreamView);

    // TODO: better window for preprocessed
    connect(patternMiner, &FacePatternMiner::preprocessed, this, &FacialRecognition::_updateTrainingStreamView);

    // connect preprocessingTerminated signal of miner to the GUI, to show the mined positive and negative pattern
    connect(patternMiner, &FacePatternMiner::mining_terminated,[=](const cv::Mat &positive, const cv::Mat &negative) {
        this->_updatePositivePatternStreamView(positive);
        this->_updateNegativePatternStreamView(negative);
    });

    connect(patternMiner, &FacePatternMiner::built_classifier, this, [=](FaceClassifier *classifier) {
        _faceClassifier = classifier;

        cv::Mat test = cv::imread("./datasets/test.jpg");
        _faceClassifier->classify(test);
        cv::namedWindow("test1");
        cv::imshow("test1", test);

        cv::Mat test2 = cv::imread("./datasets/BioID-FaceDatabase-V1.2/BioID_0921.pgm");
        _faceClassifier->classify(test2);
        cv::namedWindow("test2");
        cv::imshow("test2", test2);

        cv::Mat test3 = cv::imread("./datasets/test2.jpg");
        _faceClassifier->classify(test3);
        cv::namedWindow("test3");
        cv::imshow("test3", test3);

        cv::Mat test4 = cv::imread("./datasets/24.jpg");
        _faceClassifier->classify(test4);
        cv::namedWindow("test4");
        cv::imshow("test4", test4);

    });

    // start the miner thread
    minerThread->start();


}

void FacialRecognition::_updateCamView(const cv::Mat& frame)
{/*
    if(_faceClassifier != NULL) {
        cv::Mat live(frame);
        _faceClassifier->classify(live);
        _getCamStreamView()->setImage(Cv2Qt::cvMatToQImage(live));
    } else { */
        _getCamStreamView()->setImage(Cv2Qt::cvMatToQImage(frame));
    //}
}

void FacialRecognition::_updateTrainingStreamView(const cv::Mat& frame) {
    _getTrainingStreamView()->setImage(Cv2Qt::cvMatToQImage(frame));
}

void FacialRecognition::_updatePositivePatternStreamView(const cv::Mat& frame) {
    _getPositivePatternStreamView()->setImage(Cv2Qt::cvMatToQImage(frame));
}

void FacialRecognition::_updateNegativePatternStreamView(const cv::Mat& frame) {
    _getNegativePatternStreamView()->setImage(Cv2Qt::cvMatToQImage(frame));
}

void FacialRecognition::_handleClick(const cv::Point& point)
{
    std::cout << point << std::endl;
}

FacialRecognition::~FacialRecognition()
{
    delete ui;
}

VideoStreamView* FacialRecognition::_getCamStreamView() {
    return reinterpret_cast<VideoStreamView*>(ui->gridLayout->itemAtPosition(0,0)->widget());
}

VideoStreamView* FacialRecognition::_getTrainingStreamView() {
    return reinterpret_cast<VideoStreamView*>(ui->gridLayout->itemAtPosition(0,2)->widget());
}

VideoStreamView* FacialRecognition::_getPositivePatternStreamView() {
    return reinterpret_cast<VideoStreamView*>(ui->gridLayout->itemAtPosition(2,0)->widget());
}

VideoStreamView* FacialRecognition::_getNegativePatternStreamView() {
    return reinterpret_cast<VideoStreamView*>(ui->gridLayout->itemAtPosition(2,2)->widget());
}
