#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = _t2 = -1;
}

void FeatureClassifier::setConstants(cv::Mat1b &raw, uint32_t *_c1, uint32_t *_c2, uint32_t *_c3, uint32_t *_c4) {
    cv::Mat1b edge = Preprocessor::edge(raw);
    *_c1 = *_c2 = *_c3 = *_c4 = 0;
    for(const cv::Point &point : _positiveMFICoordinates) {
        //c1 is the sum of pixel intesities of the positive feature pattern
        // in the raw image
        *_c1 += raw.at<uchar>(point);
        //c3 is the sum of pixel intesities of the positive faeture pattern
        // in the edge image
        *_c3 += edge.at<uchar>(point);
    }

    for(const cv::Point &point : _negativeMFICoordinates) {
        //c2 is the sum of pixel intesities of the negatie feature pattern
        // in the raw image
        *_c2 += raw.at<uchar>(point);
        //c4 is the sum of pixel intesities of the negative faeture pattern
        // in the edge image
        *_c4 += edge.at<uchar>(point);
    }
}

void FeatureClassifier::train(bool positive, QString trainingSet) {
    uint32_t _c1, _c2, _c3, _c4;
    QDirIterator *it = new QDirIterator(trainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        if(positive) {
            auto diff = _c1 - _c2;
            while(diff <= _t1) {
                --_t1;
            }
            diff = _c3 - _c4;
            while(diff <= _t2) {
                --_t2;
            }
        } else {
            auto diff = _c1 - _c2;
            while(diff > _t1) {
                ++_t1;
            }
            diff = _c3 - _c4;
            while(diff > _t2) {
                ++_t2;
            }
        }
    }
    delete it;

    std::cout << "Thresholds: " << _t1 << " " << _t2 << std::endl;
}

bool FeatureClassifier::classify(cv::Mat1b &window) {
    uint32_t _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);
    return _c1 - _c2 > _t1 && _c1 - _c2 > _t2;
}



