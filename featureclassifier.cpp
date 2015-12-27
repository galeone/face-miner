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

void FeatureClassifier::setConstants(const cv::Mat1b &gray, double *_c1, double *_c2, double *_c3, double *_c4) {
    cv::Mat1b edge = Preprocessor::equalize(gray);
    edge = Preprocessor::edge(edge);

    *_c1 = *_c2 = *_c3 = *_c4 = 0;
    for(const cv::Point &point : _positiveMFICoordinates) {
        //c1 is the sum of pixel intesities of the positive feature pattern
        // in the gray image
        *_c1 += gray.at<uchar>(point);
        //c3 is the sum of pixel intesities of the positive faeture pattern
        // in the edge image
        *_c3 += edge.at<uchar>(point);
    }

    for(const cv::Point &point : _negativeMFICoordinates) {
        //c2 is the sum of pixel intesities of the negative feature pattern
        // in the gray image
        *_c2 += gray.at<uchar>(point);
        //c4 is the sum of pixel intesities of the negative faeture pattern
        // in the edge image
        *_c4 += edge.at<uchar>(point);
    }
}

void FeatureClassifier::train(std::vector<cv::Mat1b> &truePositive, std::vector<cv::Mat1b> &falsePositive) {
    std::vector<double> positiveT1, negativeT1, positiveT2, negativeT2, positiveCoeff[4], negativeCoeff[4];
    double _c1, _c2, _c3, _c4;

    for(const auto &gray : truePositive) {
        setConstants(gray, &_c1, &_c2, &_c3, &_c4);

        positiveT1.push_back(_c1 - _c2);
        positiveT2.push_back(_c3 - _c4);

        positiveCoeff[0].push_back(_c1);
        positiveCoeff[1].push_back(_c2);
        positiveCoeff[2].push_back(_c3);
        positiveCoeff[3].push_back(_c4);
    }

    for(const auto &gray : falsePositive) {
        setConstants(gray, &_c1, &_c2, &_c3, &_c4);

        negativeT1.push_back(_c1 - _c2);
        negativeT2.push_back(_c3 - _c4);

        negativeCoeff[0].push_back(_c1);
        negativeCoeff[1].push_back(_c2);
        negativeCoeff[2].push_back(_c3);
        negativeCoeff[3].push_back(_c4);
    }

    for(auto i=0;i<4;++i) {
        _tUpper[i] = *std::max_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
        _tLower[i] = *std::min_element(positiveCoeff[i].begin(), positiveCoeff[i].end());
    }

    //_t1 = equal_error_rate(positiveT1,negativeT1).second - 255*5;
    //_t2 = equal_error_rate(positiveT2,negativeT2).second - 255*5;
    _t1 = _tLower[0] - _tUpper[1]; // c1 - c2
    _t2 = _tLower[2] - _tUpper[3]; // c3 - c4

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
        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b gray = Preprocessor::gray(raw);
        positive.push_back(gray);
    }

    delete it;

    it = new QDirIterator(negativeTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b gray = Preprocessor::gray(raw);
        negative.push_back(gray);
    }

    delete it;

    return train(positive, negative);
}

// Classify suppose a gray window
bool FeatureClassifier::classify(const cv::Mat1b &window) {
    double _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);

    return _c1 - _c2 > _t1
            && _c3 - _c4 > _t2
            && _tLower[0] < _c1 && _tUpper[0] > _c1
            && _tLower[1] < _c2 && _tUpper[1] > _c2
            && _tLower[2] < _c3 && _tUpper[2] > _c3
            && _tLower[3] < _c4 && _tUpper[3] > _c4;
}
