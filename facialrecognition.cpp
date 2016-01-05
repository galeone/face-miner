#include "facialrecognition.h"
#include "ui_facialrecognition.h"

void FacialRecognition::_startCamStream() {
  cv::VideoCapture _cam(0);
  _cam.set(CV_CAP_PROP_FRAME_WIDTH, _streamSize->width());
  _cam.set(CV_CAP_PROP_FRAME_HEIGHT, _streamSize->height());

  if (!_cam.isOpened()) {
    QMessageBox error(this);
    error.critical(this, "Error", "Unable to open webcam");
    error.show();
  } else {
    QThread* frameStreamThread = new QThread();
    CamStream* frameStream = new CamStream(_cam);

    // Move the object frameStream to his own thread
    frameStream->moveToThread(frameStreamThread);

    // sending CamStream::newFrame output, into FacialRecognition::updateCamView
    // input
    connect(frameStream, &CamStream::newFrame, this,
            &FacialRecognition::_updateCamView);

    // start frameStream on frameStreamThread start
    connect(frameStreamThread, &QThread::started, frameStream,
            &CamStream::start);

    // sending click coordinates into CamStremView to
    // FacialRecognition::_handleClick
    connect(_getCamStreamView(), &VideoStreamView::clicked, this,
            &FacialRecognition::_handleClick);

    // set CamStreamView to fixed size
    _getCamStreamView()->setSize(*_streamSize);

    // start thread: stream of frames
    frameStreamThread->start();
  }
}

FacialRecognition::FacialRecognition(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::FacialRecognition) {
  ui->setupUi(this);
  this->showMaximized();

  _streamSize = new QSize(300, 300);

  // register cv::Mat type, thus it can be used in signals and slots
  qRegisterMetaType<cv::Mat>("cv::Mat");

  // set TraingStreamView to fixed size
  _getTrainingStreamView()->setSize(*_streamSize);

  _getPositivePatternStreamView()->setSize(*_streamSize);
  _getNegativePatternStreamView()->setSize(*_streamSize);

  // Create a thread for the miner
  QThread* minerThread = new QThread();

  // Create the face pattern miner
  // TODO: make the dataset with positive and negative items selectable from the
  // view
  FacePatternMiner* patternMiner = new FacePatternMiner(
      "./datasets/mitcbcl/train/face/", "./datasets/mitcbcl/train/non-face/",
      "./datasets/mitcbcl/test/face/", "./datasets/mitcbcl/test/non-face/");

  // move the miner to his own thread
  patternMiner->moveToThread(minerThread);

  // connect signal start of the thread to the start() method of the miner
  connect(minerThread, &QThread::started, patternMiner,
          &FacePatternMiner::start);

  // connect preprocessing signal of miner to the GUI, to show what image has
  // being processed
  connect(patternMiner, &FacePatternMiner::preprocessing, this,
          &FacialRecognition::_updateTrainingStreamView);

  // TODO: better window for preprocessed
  connect(patternMiner, &FacePatternMiner::preprocessed, this,
          &FacialRecognition::_updateTrainingStreamView);

  // connect preprocessingTerminated signal of miner to the GUI, to show the
  // mined positive and negative pattern
  connect(patternMiner, &FacePatternMiner::mining_terminated,
          [=](const cv::Mat& positive, const cv::Mat& negative) {
            this->_updatePositivePatternStreamView(positive);
            this->_updateNegativePatternStreamView(negative);
          });

  connect(patternMiner, &FacePatternMiner::built_classifier, this,
          [=](FaceClassifier* classifier) {
            _faceClassifier = classifier;

            cv::Mat test2 =
                cv::imread("./datasets/BioID-FaceDatabase-V1.2/BioID_0921.pgm");
            auto Start = cv::getTickCount();
            auto faces = _faceClassifier->classify(test2);
            auto End = cv::getTickCount();
            auto seconds = (End - Start) / cv::getTickFrequency();
            std::cout << "Time: " << seconds << std::endl;
            for (const auto& face : faces) {
              cv::rectangle(test2, face, cv::Scalar(255, 255, 0));
            }
            cv::namedWindow("test2");
            cv::imshow("test2", test2);

            cv::Mat test3 = cv::imread("./datasets/test2.jpg");
            faces = _faceClassifier->classify(test3);
            for (const auto& face : faces) {
              cv::rectangle(test3, face, cv::Scalar(255, 255, 0));
            }
            cv::namedWindow("test3");
            cv::imshow("test3", test3);

            cv::Mat test4 = cv::imread("./datasets/24.jpg");
            faces = _faceClassifier->classify(test4);
            for (const auto& face : faces) {
              cv::rectangle(test4, face, cv::Scalar(255, 255, 0));
            }
            cv::namedWindow("test4");
            cv::imshow("test4", test4);

            _startCamStream();
          });

  // start the miner thread
  minerThread->start();
}

/**
 * Perform template matching to search the user's face in the given image.
 * Updates rectangles
 * @param   im    The source image
 */
void FacialRecognition::_track(const cv::Mat& im) {
  cv::Mat1b grayFrame = Preprocessor::gray(im);
  for (auto& pair : _camFaces) {
    cv::Mat1f dst;
    cv::matchTemplate(grayFrame, pair.second, dst, CV_TM_SQDIFF_NORMED);

    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

    if (minval <= 0.2) {
      pair.first.x = minloc.x;
      pair.first.y = minloc.y;
    } else {
      pair.first.x = pair.first.y = pair.first.width = pair.first.height = 0;
      _camFaces.clear();
    }
  }
}

void FacialRecognition::_updateCamView(const cv::Mat& frame) {
  if (_camFaces.size() == 0) {
    auto faces = _faceClassifier->classify(frame);
    if (faces.size() > 0) {
      _camFaces.reserve(faces.size());
      for (const auto& faceRect : faces) {
        cv::Mat1b grayFrame = Preprocessor::gray(frame);
        cv::Mat1b faceTpl = Preprocessor::equalize(grayFrame(faceRect));
        _camFaces.push_back(std::make_pair(faceRect, faceTpl));
      }
    }
  }

  if (_camFaces.size() > 0) {
    _track(frame);
  }

  cv::Mat frame2;
  frame.copyTo(frame2);

  for (const auto& pair : _camFaces) {
    cv::rectangle(frame2, pair.first, cv::Scalar(255, 255, 0));
  }

  _getCamStreamView()->setImage(Cv2Qt::cvMatToQImage(frame2));
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

void FacialRecognition::_handleClick(const cv::Point& point) {
  std::cout << point << std::endl;
}

FacialRecognition::~FacialRecognition() {
  delete ui;
}

VideoStreamView* FacialRecognition::_getCamStreamView() {
  return reinterpret_cast<VideoStreamView*>(
      ui->gridLayout->itemAtPosition(0, 0)->widget());
}

VideoStreamView* FacialRecognition::_getTrainingStreamView() {
  return reinterpret_cast<VideoStreamView*>(
      ui->gridLayout->itemAtPosition(0, 2)->widget());
}

VideoStreamView* FacialRecognition::_getPositivePatternStreamView() {
  return reinterpret_cast<VideoStreamView*>(
      ui->gridLayout->itemAtPosition(2, 0)->widget());
}

VideoStreamView* FacialRecognition::_getNegativePatternStreamView() {
  return reinterpret_cast<VideoStreamView*>(
      ui->gridLayout->itemAtPosition(2, 2)->widget());
}
