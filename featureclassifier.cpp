#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates,
                                     QString test_positive, QString test_negative) {
    _testPositive = test_positive;
    _testNegative = test_negative;
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = 0;
    _t2 = 0;

    for(auto i=0;i<4;++i) {
        _tLower[i] = 0;
        _tUpper[i] = 0;
    }
}

void FeatureClassifier::setConstants(cv::Mat1b &raw, double *_c1, double *_c2, double *_c3, double *_c4) {
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
    std::vector<double> positiveT1, negativeT1, positiveT2, negativeT2, positiveCoeff[4], negativeCoeff[4];
    QDirIterator *it = new QDirIterator(positiveTrainingSet);
    auto positiveCount = 0;
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }
        ++positiveCount;
    }

    auto negativeCount = 0;
    it = new QDirIterator(negativeTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }
        ++negativeCount;
    }

    double _c1, _c2, _c3, _c4;

    it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        raw = Preprocessor::gray(raw);
        raw = Preprocessor::equalize(raw);
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        positiveT1.push_back(_c1 - _c2);
        positiveT2.push_back(_c3 - _c4);

        positiveCoeff[0].push_back(_c1 );
        positiveCoeff[1].push_back(_c2 );
        positiveCoeff[2].push_back(_c3 );
        positiveCoeff[3].push_back(_c4 );

    }

    it = new QDirIterator(negativeTrainingSet);

    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        raw = Preprocessor::gray(raw);
        raw = Preprocessor::equalize(raw);
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        negativeT1.push_back(_c1 - _c2);
        negativeT2.push_back(_c3 - _c4);

        negativeCoeff[0].push_back(_c1 );
        negativeCoeff[1].push_back(_c2 );
        negativeCoeff[2].push_back(_c3 );
        negativeCoeff[3].push_back(_c4 );
    }

    _t1 = equal_error_rate(positiveT1,negativeT1).second*2;
    _t2 = equal_error_rate(positiveT2,negativeT2).second/2;

    for(auto i=0;i<4;++i) {
        //_tUpper[i] = (*std::max_element(positiveCoeff[i].begin(), positiveCoeff[i].end()) + *std::max_element(negativeCoeff[i].begin(), negativeCoeff[i].end())) / 2;
        //_tLower[i] = (*std::min_element(positiveCoeff[i].begin(), positiveCoeff[i].end()) + *std::min_element(negativeCoeff[i].begin(), negativeCoeff[i].end())) / 2;
        _tUpper[i] = *std::max_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
        _tLower[i] = *std::min_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
    }

    std::cout << "[!] Features classfier:\n";
    std::cout << "T1: " << _t1 <<"\nT2: " << _t2 << "\n";
    for(auto i=0;i<4;++i) {
        std::cout << "T_lower{" << i << "} = " << _tLower[i] << "\n";
        std::cout << "T_upper{" << i << "} = " << _tUpper[i] << "\n";
    }
    std::cout << std::endl;

    delete it;

    Stats::print(_testPositive, _testNegative, this);
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
