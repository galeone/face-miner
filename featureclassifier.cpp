#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = _t2 = 0;
}

void FeatureClassifier::setConstants(cv::Mat1b &raw, int32_t *_c1, int32_t *_c2, int32_t *_c3, int32_t *_c4) {
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

void FeatureClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    int32_t _c1, _c2, _c3, _c4;
    std::vector<double> positiveT1, negativeT1, positiveT2, negativeT2;

    QDirIterator *it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        positiveT1.push_back(_c1 - _c2);
        positiveT2.push_back(_c3 - _c4);
    }

    it = new QDirIterator(negativeTrainingSet);

    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        negativeT1.push_back(_c1 - _c2);
        negativeT2.push_back(_c3 - _c4);
    }

    delete it;

    _t1 = equal_error_rate(positiveT1, negativeT1).second;
    _t2 = equal_error_rate(positiveT2, negativeT2).second;

    std::cout << "Thresholds: " << _t1 << " " << _t2 << std::endl;
}

bool FeatureClassifier::classify(cv::Mat1b &window) {
    int32_t _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);
    return _c1 - _c2 > _t1 && _c1 - _c2 > _t2;
}
