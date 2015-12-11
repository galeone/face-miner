#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = new cv::Boost();
    _t2 = new cv::Boost();
}

void FeatureClassifier::setConstants(cv::Mat1b &raw, int32_t *_c1, int32_t *_c2, int32_t *_c3, int32_t *_c4) {
    cv::Mat1b edge = Preprocessor::gray(raw);
    edge = Preprocessor::equalize(raw);
    edge = Preprocessor::edge(raw);
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

    cv::Mat1f labelsT1(positiveCount + negativeCount,1,CV_32FC1),
            samplesT1(positiveCount + negativeCount,1, CV_32FC1),
            labelsT2(positiveCount + negativeCount,1, CV_32FC1),
            samplesT2(positiveCount + negativeCount,1, CV_32FC1);

    auto counter = 0;

    int32_t _c1, _c2, _c3, _c4;

    it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        labelsT1.at<float>(counter, 0) = 1;
        samplesT1.at<float>(counter, 0) = _c1 - _c2;

        labelsT2.at<float>(counter, 0) = 1;
        samplesT2.at<float>(counter, 0) = _c3 - _c4;
        ++counter;
    }

    it = new QDirIterator(negativeTrainingSet);

    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat1b raw = cv::imread(fileName.toStdString());
        setConstants(raw, &_c1, &_c2, &_c3, &_c4);

        labelsT1.at<float>(counter, 0) = -1;
        samplesT1.at<float>(counter, 0) = _c1 - _c2;

        labelsT2.at<float>(counter, 0) = -1;
        samplesT2.at<float>(counter, 0) = _c3 - _c4;
        ++counter;
    }

    delete it;

    cv::Mat vartypeT1(samplesT1.cols+1,1,CV_8U);
    vartypeT1.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartypeT1.at<uchar>(samplesT1.cols,0) = CV_VAR_CATEGORICAL;

    float priors[] = { 1.0f, 1.0f };

    _t1->train(samplesT1, CV_ROW_SAMPLE, labelsT1, cv::Mat(), cv::Mat(), vartypeT1, cv::Mat(), cv::BoostParams(
                  cv::Boost::REAL,
                  100,
                  0.0,
                  1,
                  false,
                  priors));

    cv::Mat vartypeT2(samplesT2.cols+1,1,CV_8U);
    vartypeT2.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartypeT2.at<uchar>(samplesT2.cols,0) = CV_VAR_CATEGORICAL;

    _t2->train(samplesT2, CV_ROW_SAMPLE, labelsT2, cv::Mat(), cv::Mat(), vartypeT2, cv::Mat(), cv::BoostParams(
                  cv::Boost::REAL,
                  100,
                  0.0,
                  1,
                  false,
                  priors));
}

bool FeatureClassifier::classify(cv::Mat1b &window) {
    int32_t _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);
    cv::Mat1f sample1 = (cv::Mat1f(1,1) << _c1 - _c2), sample2 = (cv::Mat1f(1,1) << _c3 - _c4);
    // TODO: altre soglie
    return _t1->predict(sample1) > 0 && _t1->predict(sample2) > 0;
}
