#include "featureclassifier.h"

FeatureClassifier::FeatureClassifier(std::vector<cv::Point> &positiveMFICoordinates,
                                     std::vector<cv::Point> &negativeMFICoordinates) {
    _positiveMFICoordinates = positiveMFICoordinates;
    _negativeMFICoordinates = negativeMFICoordinates;
    _t1 = new cv::Boost();
    _t2 = new cv::Boost();
    _delta = 300;

    for(auto i=0;i<4;++i) {
        _tLower[i] = new cv::Boost();
        _tUpper[i] = new cv::Boost();
    }
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

    auto totSamples = positiveCount + negativeCount;

    cv::Mat1f labelsT1(totSamples,1,CV_32FC1),
            samplesT1(totSamples,1, CV_32FC1),
            labelsT2(totSamples,1, CV_32FC1),
            samplesT2(totSamples,1, CV_32FC1);

    cv::Mat1f labelsUpper[4], labelsLower[4], samplesUpper[4], samplesLower[4];
    for(auto i=0;i<4;++i) {
        labelsUpper[i] = cv::Mat(totSamples, 1, CV_32FC1);
        labelsLower[i] = cv::Mat(totSamples, 1, CV_32FC1);

        samplesUpper[i] = cv::Mat(totSamples, 1, CV_32FC1);
        samplesLower[i] = cv::Mat(totSamples, 1, CV_32FC1);
    }

    auto counter = 0;

    int32_t _c1, _c2, _c3, _c4;

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

        labelsT1.at<float>(counter, 0) = 1;
        labelsT2.at<float>(counter, 0) = 1;

        samplesT1.at<float>(counter, 0) = _c1 - _c2;
        samplesT2.at<float>(counter, 0) = _c3 - _c4;

        for(auto i=0;i<4;++i) {
            labelsLower[i].at<float>(counter, 0) = 1;
            labelsUpper[i].at<float>(counter, 0) = 1;
        }

        samplesLower[0].at<float>(counter, 0) = _c1 - _delta;
        samplesUpper[0].at<float>(counter, 0) = _c1 + _delta;

        samplesLower[1].at<float>(counter, 0) = _c2 - _delta;
        samplesUpper[1].at<float>(counter, 0) = _c2 + _delta;

        samplesLower[2].at<float>(counter, 0) = _c3 - _delta;
        samplesUpper[2].at<float>(counter, 0) = _c3 + _delta;

        samplesLower[3].at<float>(counter, 0) = _c4 - _delta;
        samplesUpper[3].at<float>(counter, 0) = _c4 + _delta;

        ++counter;
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

        labelsT1.at<float>(counter, 0) = -1;
        labelsT2.at<float>(counter, 0) = -1;

        samplesT1.at<float>(counter, 0) = _c1 - _c2;
        samplesT2.at<float>(counter, 0) = _c3 - _c4;

        for(auto i=0;i<4;++i) {
            labelsLower[i].at<float>(counter, 0) = -1;
            labelsUpper[i].at<float>(counter, 0) = -1;
        }

        samplesLower[0].at<float>(counter, 0) = _c1 - _delta;
        samplesUpper[0].at<float>(counter, 0) = _c1 + _delta;

        samplesLower[1].at<float>(counter, 0) = _c2 - _delta;
        samplesUpper[1].at<float>(counter, 0) = _c2 + _delta;

        samplesLower[2].at<float>(counter, 0) = _c3 - _delta;
        samplesUpper[2].at<float>(counter, 0) = _c3 + _delta;

        samplesLower[3].at<float>(counter, 0) = _c4 - _delta;
        samplesUpper[3].at<float>(counter, 0) = _c4 + _delta;
        ++counter;
    }

    delete it;

    cv::Mat vartypeT1(samplesT1.cols+1,1,CV_8U);
    vartypeT1.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartypeT1.at<uchar>(samplesT1.cols,0) = CV_VAR_CATEGORICAL;

    float priors[] = { 1.0f, 1.0f };

    _t1->train(samplesT1, CV_ROW_SAMPLE, labelsT1, cv::Mat(), cv::Mat(), vartypeT1, cv::Mat(), cv::BoostParams(
                   cv::Boost::REAL, // boost type
                   300, // weak count
                   0.95, // weight trim rate
                   25, // max depth
                   false, // use surrogates
                   priors));

    cv::Mat vartypeT2(samplesT2.cols+1,1,CV_8U);
    vartypeT2.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartypeT2.at<uchar>(samplesT2.cols,0) = CV_VAR_CATEGORICAL;

    _t2->train(samplesT2, CV_ROW_SAMPLE, labelsT2, cv::Mat(), cv::Mat(), vartypeT2, cv::Mat(), cv::BoostParams(
                   cv::Boost::REAL, // boost type
                   300, // weak count
                   0.95, // weight trim rate
                   25, // max depth
                   false, // use surrogates
                   priors));


    for(auto i=0;i<4;++i) {
        cv::Mat vartype(samplesLower[i].cols+1,1,CV_8U);
        vartype.setTo(cv::Scalar(CV_VAR_NUMERICAL));
        vartype.at<uchar>(samplesT1.cols,0) = CV_VAR_CATEGORICAL;

        _tLower[i]->train(samplesLower[i], CV_ROW_SAMPLE, labelsLower[i], cv::Mat(), cv::Mat(), vartype, cv::Mat(), cv::BoostParams(
                              cv::Boost::REAL, // boost type
                              300, // weak count
                              0.95, // weight trim rate
                              25, // max depth
                              false, // use surrogates
                              priors));

        _tUpper[i]->train(samplesUpper[i], CV_ROW_SAMPLE, labelsUpper[i], cv::Mat(), cv::Mat(), vartype, cv::Mat(), cv::BoostParams(
                              cv::Boost::REAL, // boost type
                              300, // weak count
                              0.95, // weight trim rate
                              25, // max depth
                              false, // use surrogates
                              priors));
    }
}

// Classify suppose gray and equalized window
bool FeatureClassifier::classify(cv::Mat1b &window) {
    int32_t _c1, _c2, _c3, _c4;
    setConstants(window, &_c1, &_c2, &_c3, &_c4);
    cv::Mat1f sample1 = (cv::Mat1f(1,1) << _c1 - _c2),
            sample2 = (cv::Mat1f(1,1) << _c3 - _c4),
            // samples for rule 3
            sampleR31 = (cv::Mat1f(1,1) << _c1),
            sampleR32 = (cv::Mat1f(1,1) << _c2),
            sampleR33 = (cv::Mat1f(1,1) << _c3),
            sampleR34 = (cv::Mat1f(1,1) << _c4);

    return _t1->predict(sample1) > 0
            && _t2->predict(sample2) > 0
            && _tLower[0]->predict(sampleR31) > 0
            && _tLower[1]->predict(sampleR32) > 0
            && _tLower[2]->predict(sampleR33) > 0
            && _tLower[3]->predict(sampleR34) > 0;
}
