#include "svmclassifier.h"

SVMClassifier::SVMClassifier(const cv::Range &rows1, const cv::Range rows2)
{
    _r1 = rows1;
    _r2 = rows2;
    _svm = new cv::SVM();
}

void SVMClassifier::_getIntegralImage() {

}

void SVMClassifier::_getHaarCoefficients(cv::Mat1b &window, cv::Mat1f &coeff) {
    // filter2d with haar kernel
    // extract coeff
    // put coefficients into coeff
    //coeff =  (cv::Mat1f(1,1) << t);
}

bool SVMClassifier::classify(cv::Mat1b &window) { /*
    cv::Mat1f coeff;
    _getHaarCoefficients(window, &coeff);

    auto predictedLabel = _svm->predict(coeff);

    if(predictedLabel < 0 ) { // non face threshold
        return false;
    }
    return true;
    */
}

void SVMClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    /*
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

    cv::Mat1f labels(positiveCount + negativeCount,1,CV_32FC1),
            samples(positiveCount + negativeCount,1, CV_32FC1);

    auto counter = 0;

    it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat face = cv::imread(fileName.toStdString());
        face = Preprocessor::gray(face);

        // get haar coefficents
        _getHaarCoefficients(face, row);

        labels.at<float>(counter, 0) = 1;
        // loop per ogni coefficiente e inserisci al posto giusto
        //samples.at<float>(counter, 0) = t;
        ++counter;

    }

    it = new QDirIterator(negativeTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat face = cv::imread(fileName.toStdString());
        face = Preprocessor::gray(face);

        // get haar coefficents
        _getHaarCoefficients(face, row);

        labels.at<float>(counter, 0) = -1;
        // loop per ogni coefficiente e inserisci al posto giusto (samples(coutner, i))

        ++counter;
    }

    delete it;

    cv::Mat vartype(samples.cols+1,1,CV_8U);
    vartype.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    vartype.at<uchar>(samples.cols,0) = CV_VAR_CATEGORICAL;

    float priors[] = { 1.0f, 1.0f };

    _svm->train(samples, labels, cv::Mat(), cv::Mat(),cv::SVMParams(
                  cv::SVM::C_SVC,
                  cv::SVM::POLY,
                  3,
                  1,
                  false,
                  priors));
                  */
}
