#include "facefinder.h"

// slot
void FaceFinder::setClassifier(FaceClassifier* fc) {
  _fc = fc;
  emit ready();
}

// emit signal found if faces are found
std::vector<std::pair<cv::Rect, cv::Mat1b>> FaceFinder::find(const cv::Mat& frame) {
  auto faces = _fc->classify(frame);
  std::vector<std::pair<cv::Rect, cv::Mat1b>> _faces;
  if (faces.size() > 0) {
    _faces.reserve(faces.size());
    for (const auto& faceRect : faces) {
      cv::Mat1b grayFrame = Preprocessor::gray(frame);
      cv::Mat1b faceTpl = Preprocessor::equalize(grayFrame(faceRect));
      _faces.push_back(std::make_pair(faceRect, faceTpl));
    }
    emit found(_faces);
  }
  // return for syncronous call
  return _faces;
}
