#include "faceclassifier.h"

FaceClassifier::FaceClassifier(VarianceClassifier* vc,
                               FeatureClassifier* fc,
                               SVMClassifier* svmc,
                               cv::Size size) {
  _vc = vc;
  _fc = fc;
  _sc = svmc;
  _windowSize = size;
  _scaleFactor = 1.25;
  _step = 2;
}

// Returns true if contains some faces. Hilight with a rectangle the face on the
// image.
std::vector<cv::Rect> FaceClassifier::classify(const cv::Mat& image) {
  std::vector<cv::Rect> allCandidates;
  cv::Mat1b gray = Preprocessor::gray(image);

  // pyramid downsampling
  // from smaller to bigger.
  // avoid to search where a face in a lower scale is found < image, scale
  // factor >
  std::vector<std::pair<cv::Mat1b, float>> pyramid;
  for (float factor = 1;; factor *= _scaleFactor) {
    // Size of the image scaled down (from bigger to smaller)
    cv::Size sz(image.cols / factor, image.rows / factor);

    // Difference between sized of the scaled image and the original detection
    // window
    cv::Size sz1(sz.width - _windowSize.width, sz.height - _windowSize.height);

    // if the actual scaled image is smaller than the origina detection window,
    // break
    if (sz1.width < 0 || sz1.height < 0) {
      break;
    }

    cv::Mat1b level;
    cv::resize(gray, level, sz, 0, 0, cv::INTER_NEAREST);
    pyramid.push_back(std::pair<cv::Mat1b, float>(level, factor));
  }

  // from smaller to bigger
  for (auto rit = pyramid.rbegin(); rit != pyramid.rend(); rit++) {
    cv::Mat1b level = (*rit).first;
    float factor = (*rit).second;
    _slidingSearch(level, factor, allCandidates);
  }

  return allCandidates;
}

// allCandidates contains the previous positions, scaled to the original
// dimension of image of the found face
// thus we can exploit this thing to skip a big section of the level = the
// intersection of the candidates
// scaled by factor (thus is compatible with level) with level
void FaceClassifier::_slidingSearch(cv::Mat1b& level,
                                    float factor,
                                    std::vector<cv::Rect>& allCandidates) {
  cv::Size winSize(std::ceil(_windowSize.width * factor),
                   std::ceil(_windowSize.height * factor));

  std::vector<cv::Rect> toSkip;
  for (const cv::Rect& r : allCandidates) {
    toSkip.push_back(cv::Rect(r.x / factor, r.y / factor, r.width / factor,
                              r.height / factor));
  }

  for (auto y = 0; y < level.rows - _windowSize.height; y += _step) {
    for (auto x = 0; x < level.cols - _windowSize.width; x += _step) {
      cv::Rect roi_rect(x, y, _windowSize.width, _windowSize.height);

      // if roi_rect intersect a toSkip element, lets continue
      auto exists =
          std::find_if(toSkip.begin(), toSkip.end(), [&](const cv::Rect& skip) {
            cv::Rect intersection(skip & roi_rect);
            if (intersection.area() > 0) {
              x += intersection.width + _step;
              return true;
            }
            return false;
          });

      if (exists != toSkip.end()) {  // intersection exists, we can skip
        continue;
      }

      cv::Mat1b roi = level(roi_rect);
      if (_vc->classify(roi)) {  // variance
        // std::cout << "V";
        if (_fc->classify(roi)) {  // features (shape)
          // std::cout << "F";
          if (_sc->classify(roi)) {  // svm to refine
            // std::cout << "S";
            cv::Rect destPos(std::floor(x * factor), std::floor(y * factor),
                             winSize.width + 1, winSize.height + 1);
            allCandidates.push_back(destPos);
            // add current roi to toSkip vector
            toSkip.push_back(roi_rect);
            x += _windowSize.height;
          }
        }
        // std::cout << std::endl;
      }
    }
  }
}
