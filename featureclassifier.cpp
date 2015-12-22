#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = 0;
    _t2 = 0;

    for(auto i=0;i<4;++i) {
        _tLower[i] = 0;
        _tUpper[i] = 0;
    }
}

void FeatureClassifier::setConstants(const cv::Mat1b &raw, double *_c1, double *_c2, double *_c3, double *_c4) {
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

// images are just equalized
void FeatureClassifier::train(std::vector<cv::Mat1b> &truePositive, std::vector<cv::Mat1b> &falsePositive) {
    std::vector<double> positiveT1, negativeT1, positiveT2, negativeT2, positiveCoeff[4], negativeCoeff[4];
    double _c1, _c2, _c3, _c4;

    for(const auto &raw : truePositive) {
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);
        setConstants(face, &_c1, &_c2, &_c3, &_c4);

        positiveT1.push_back(_c1 - _c2);
        positiveT2.push_back(_c3 - _c4);

        positiveCoeff[0].push_back(_c1);
        positiveCoeff[1].push_back(_c2);
        positiveCoeff[2].push_back(_c3);
        positiveCoeff[3].push_back(_c4);
    }

    for(const auto &raw : falsePositive) {
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);
        setConstants(face, &_c1, &_c2, &_c3, &_c4);

        negativeT1.push_back(_c1 - _c2);
        negativeT2.push_back(_c3 - _c4);

        negativeCoeff[0].push_back(_c1 );
        negativeCoeff[1].push_back(_c2 );
        negativeCoeff[2].push_back(_c3 );
        negativeCoeff[3].push_back(_c4 );
    }

    _t1 = equal_error_rate(positiveT1,negativeT1).second*2.5;
    _t2 = equal_error_rate(positiveT2,negativeT2).second/2.5;

    for(auto i=0;i<4;++i) {
        _tUpper[i] = *std::max_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
        _tLower[i] = *std::min_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
    }
    //_tUpper[0] non influisce
    _tLower[0] -= 255*4;

    _tUpper[1] += 255*5;
    //_tLower[1] non influisce

    _tUpper[2] += 255;
    //_tLower[2] non influisce

    //_tUpper[3] non influisce
    //_tLower[3] non influisce

    std::cout << "T1: " << _t1 <<"\nT2: " << _t2 << "\n";
    for(auto i=0;i<4;++i) {
        std::cout << "T_lower{" << i << "} = " << _tLower[i] << "\n";
        std::cout << "T_upper{" << i << "} = " << _tUpper[i] << "\n";
    }
    std::cout << std::endl;
}

void FeatureClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    QDirIterator *it = new QDirIterator(positiveTrainingSet);
    std::vector<cv::Mat1b> positive, negative;
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }
        cv::Mat1b raw = cv::imread(fileName.toStdString());
        positive.push_back(raw);
    }

    delete it;

    it = new QDirIterator(negativeTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        negative.push_back(raw);
    }

    delete it;

    return train(positive, negative);
}

// Classify suppose gray and equalized window
bool FeatureClassifier::classify(cv::Mat1b &window) {
    double _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);

    return _c1 - _c2 > _t1
            && _c3 - _c4 > _t2
            && _tLower[0] < _c1 && _tUpper[0] > _c1
            && _tLower[1] < _c2 && _tUpper[1] > _c2
            && _tLower[2] < _c3 && _tUpper[2] > _c3
            && _tLower[3] < _c4 && _tUpper[3] > _c4;
}
