#include "facialrecognition.h"
#include "ui_facialrecognition.h"

void FacialRecognition::_startCamStream() {
  cv::VideoCapture _cam(0);

  _frameCount = 0;

  // Register std::vector<std::pair<cv::Rect,cv::Mat1b> >& thus it can be used
  // in signals and slots
  qRegisterMetaType<std::vector<std::pair<cv::Rect, cv::Mat1b>>>(
      "std::vector<std::pair<cv::Rect,cv::Mat1b>>");

  if (!_cam.isOpened()) {
    QMessageBox error(this);
    error.critical(this, "Error", "Unable to open webcam");
    error.show();
  } else {
    std::cout << "Device resolution: " << _cam.get(CV_CAP_PROP_FRAME_WIDTH)
              << "x" << _cam.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;

    QThread* frameStreamThread = new QThread();
    _frameStream = new CamStream(_cam);

    // Move the object frameStream to his own thread
    _frameStream->moveToThread(frameStreamThread);

    // _updateCamView is the default visualizer until some faces are found.
    // when a face is found, that signal is disconnected and _track visualizes
    // the tacked faces
    // every time the background worker found a different number of faces, track
    // is reinvoked.
    // when track loses a face picture (0 rectangles), updateCamView is resetted
    // up as default.

    // sending CamStream::newFrame output to FaceFinder::find input and to
    // _track to display and track
    connect(_frameStream, &CamStream::newFrame, _faceFinder,
            [&](const cv::Mat& frame) {
              if (_camFaces.size() == 0 &&
                  (_frameCount >= 50 || _frameCount == 0)) {
                cv::Mat frame2;
                _faceFinder->find(frame);
                _frameCount = 1;
              }
              _frameCount++;
              _track(frame);
            });

    // start _frameStream on frameStreamThread start
    connect(frameStreamThread, &QThread::started, _frameStream,
            &CamStream::start);

    connect(_faceFinder, &FaceFinder::found, this,
            [&](std::vector<std::pair<cv::Rect, cv::Mat1b>> v) {
              _camFaces.clear();
              _camFaces.reserve(v.size());
              _camFaces.insert(_camFaces.end(), v.begin(), v.end());
            });

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
  // TODO: make the dataset with positive and negative items selectable from
  // the
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

  QThread* faceFinderThread = new QThread();
  _faceFinder = new FaceFinder();
  _faceFinder->moveToThread(faceFinderThread);

  // passing built classifier to the face finder
  connect(patternMiner, &FacePatternMiner::built_classifier, _faceFinder,
          &FaceFinder::setClassifier);

  // start the miner thread
  minerThread->start();

  // start the FaceFinder thread
  faceFinderThread->start();

  // handle faceFinder::ready signal
  connect(_faceFinder, &FaceFinder::ready, this, [&]() {
    auto i = 1;

    std::vector<std::string> paths;
    paths.reserve(1600);
    QDirIterator* it = new QDirIterator(QString("./datasets/yalefaces/"));
    while (it->hasNext()) {
      auto file = it->next();
      if (Preprocessor::validMime(file)) {
        paths.push_back(file.toStdString());
      }
    }
    double totalTime = 0;
    size_t detectedFaces = 0;

    // sync execution
    for (const std::string& path : paths) {
      std::cout << path << ": ";
      cv::Mat test = cv::imread(path);
      auto Start = cv::getTickCount();
      auto faces = _faceFinder->find(test);
      auto End = cv::getTickCount();
      auto seconds = (End - Start) / cv::getTickFrequency();
      totalTime += seconds;
      ++i;
      std::cout << "Time: " << seconds << "s (" << faces.size() << ") "
                << test.size() << std::endl;
      for (const auto& face : faces) {
        cv::rectangle(test, face.first, cv::Scalar(255, 255, 0));
      }
      detectedFaces += faces.size();
      std::string name = "test" + std::to_string(i);
      cv::namedWindow(name);
      cv::imshow(name, test);

    }
    std::cout << "[!] Tested using yalefaces\n";
    std::cout << "[!] Processed " << i << " images\n";
    std::cout << "[!] Detected " << detectedFaces << " faces\n";
    std::cout << "[!] Elapsed time: " << totalTime << "\n";
    std::cout << "[!] Average time: " << totalTime / i << std::endl;
    /*
    std::cout << "[!] True positive: " << tp << std::endl;
    std::cout << "[!] True negative: " << tn << std::endl;
    std::cout << "[!] False positive: " << fp << std::endl;
    std::cout << "[!] False negative: " << fn << std::endl;
    */
    //_startCamStream();
  });
}

void FacialRecognition::_track(const cv::Mat& frame) {
  cv::Mat1b grayFrame = Preprocessor::gray(frame);
  bool noneTracked = true;
  for (auto& pair : _camFaces) {
    cv::Mat1f dst;
    cv::matchTemplate(grayFrame, pair.second, dst, CV_TM_SQDIFF_NORMED);

    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

    if (minval <= 0.2) {
      pair.first.x = minloc.x;
      pair.first.y = minloc.y;
      noneTracked = false;
    } else {
      pair.first.x = pair.first.y = pair.first.width = pair.first.height = 0;
    }
  }
  // if tracking fails, updateCamView re-become the default vievew thus
  // disconnect _track and reconnect updateCamView
  if (noneTracked) {
    _camFaces.clear();
    _updateCamView(frame);
  } else {
    cv::Mat frame2;
    frame.copyTo(frame2);

    for (const auto& pair : _camFaces) {
      cv::rectangle(frame2, pair.first, cv::Scalar(255, 255, 0), 2);
    }
    _updateCamView(frame2);
  }
}

void FacialRecognition::_updateCamView(const cv::Mat& frame) {
  _getCamStreamView()->setImage(Cv2Qt::cvMatToQImage(frame));
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
