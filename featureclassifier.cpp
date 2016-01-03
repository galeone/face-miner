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

void FeatureClassifier::_setConstants(const cv::Mat1b &gray, float *_c1, float *_c2, float *_c3, float *_c4) {
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
    std::vector<float> positiveT1, positiveT2, positiveCoeff[4];
    float _c1, _c2, _c3, _c4;

    for(const auto &gray : truePositive) {
        _setConstants(gray, &_c1, &_c2, &_c3, &_c4);

        positiveT1.push_back(_c1 - _c2);
        positiveT2.push_back(_c3 - _c4);

        positiveCoeff[0].push_back(_c1);
        positiveCoeff[1].push_back(_c2);
        positiveCoeff[2].push_back(_c3);
        positiveCoeff[3].push_back(_c4);
    }


    for(auto i=0;i<4;++i) {
        std::sort(positiveCoeff[i].begin(), positiveCoeff[i].end());
        positiveCoeff[i].erase(std::unique(positiveCoeff[i].begin(), positiveCoeff[i].end()),positiveCoeff[i].end());
        size_t size = positiveCoeff[i].size();
        size_t elm = size/16;
        std::cout << "No duplicates: " << size << "\n";
        _tLower[i] = std::accumulate(positiveCoeff[i].begin(), positiveCoeff[i].begin()+elm, 0.0f)/(float)elm;
        _tUpper[i] = std::accumulate(positiveCoeff[i].end() - elm, positiveCoeff[i].end(), 0.0f)/(float)elm;
    }

    std::sort(positiveT1.begin(), positiveT1.end());
    positiveT1.erase(std::unique(positiveT1.begin(), positiveT1.end()),positiveT1.end());
    size_t size = positiveT1.size();
    std::cout << "Positive T1 size: " << size << "\n";
    size_t elm = size/12;
    _t1 = std::accumulate(positiveT1.begin(),positiveT1.begin()+elm, 0.0f)/(float)elm;

    std::sort(positiveT2.begin(), positiveT2.end());
    positiveT2.erase(std::unique(positiveT2.begin(), positiveT2.end()),positiveT2.end());
    size = positiveT2.size();
    std::cout << "Positive T2 size: " << size << "\n";
    elm = size/12;
    _t2 = std::accumulate(positiveT2.begin(),positiveT2.begin()+elm, 0.0f)/(float)elm;

    _t1 -= 255;
    _t2 -= 255;
    //_t1 -= 255*2;
    //_t2 -= 255*3;

    _tLower[0] += 255*2;
    _tUpper[0] -= 255*2;

    _tLower[1] += 255;
    _tUpper[1] -= 255;

    _tLower[2] += 255;
    _tUpper[2] -= 255*2;

    _tLower[3] += 255;
    _tUpper[3] -= 255;

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
bool FeatureClassifier::classify(const cv::Mat1b &window, float *_c1, float *_c2, float *_c3, float *_c4) {
    _setConstants(window, _c1, _c2, _c3, _c4);

    return *_c1 - *_c2 > _t1
            && *_c3 - *_c4 > _t2
            && _tLower[0] < *_c1 && _tUpper[0] > *_c1
            && _tLower[1] < *_c2 && _tUpper[1] > *_c2
            && _tLower[2] < *_c3 && _tUpper[2] > *_c3
            && _tLower[3] < *_c4 && _tUpper[3] > *_c4;
}
bool FeatureClassifier::classify(const cv::Mat1b &window) {
    float a,b,c,d;
    return classify(window,&a,&b,&c,&d);
}

