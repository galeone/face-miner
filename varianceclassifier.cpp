#include "varianceclassifier.h"

VarianceClassifier::VarianceClassifier(const cv::Size windowSize) {
    auto cols = windowSize.width,
            rows = windowSize.height;

    auto aThirdRows = std::floor(rows/3), //6
            aThirdCols = std::floor(cols/3); //6

    // Mouth region
    _E = cv::Rect(0, rows-aThirdRows, cols, aThirdRows);

    // Nose region
    _D = cv::Rect(0, rows-2*aThirdRows + 1, cols, aThirdRows-1);

    // Left eye region
    auto ac_cols = aThirdCols + 1;
    auto b_cols = cols - 2*ac_cols;

    // Top section
    if(rows % 2 != 0) {
        ++aThirdRows;
    }
    int topHeight = std::floor(aThirdRows/3);
    _A = cv::Rect(0, 2*topHeight, ac_cols, topHeight);

    // Between eye region
    _B = cv::Rect(ac_cols, 2*topHeight, b_cols, topHeight);

    // Right eye region
    _C = cv::Rect(cols - ac_cols, 2*topHeight, ac_cols, topHeight);

    _k = 1;
    /*T: << 3917.79
K: 1
True positive: 114
True negative: 15524
False positive: 8049
False negatve: 358
Precision: 0.0139655
Recall: 0.241525*/
}

cv::Scalar VarianceClassifier::_getMForABC(cv::Mat1b &window) {
    cv::Scalar mu_a, mu_b, mu_c;
    cv::Mat1b roi_a = window(_A), roi_b = window(_B), roi_c = window(_C);
    mu_a = cv::mean(roi_a);
    mu_b = cv::mean(roi_b);
    mu_c = cv::mean(roi_c);

    float ma = 0, mb = 0, mc = 0;

    uint32_t validPx = 0;

    //ma is the average intensity of those pixels that are
    // darker than the average intensity in region A
    cv::Point coord;
    for(auto x=0;x<roi_a.cols; ++x) {
        for(auto y=0;y<roi_a.rows;++y) {
            coord.x = x; coord.y = y;
            auto pxBrightness = roi_a.at<uchar>(coord);
            if(pxBrightness < mu_a[0]) {
                ma += pxBrightness;
                ++validPx;
            }
        }
    }

    ma /= (validPx > 0 ? validPx : 1);
    validPx = 0;

    //mc is the average intensity of those pixels that are
    // darker than the average intensity in region C
    for(auto x=0;x<roi_c.cols; ++x) {
        for(auto y=0;y<roi_c.rows;++y) {
            coord.x = x; coord.y = y;
            auto pxBrightness = roi_c.at<uchar>(coord);
            if(pxBrightness < mu_c[0]) {
                mc += pxBrightness;
                ++validPx;
            }
        }
    }

    mb /= (validPx > 0 ? validPx : 1);
    validPx = 0;

    //mb is the average intensity of those pixels that are
    // birghter than the average intensity in region B
    for(auto x=0;x<roi_b.cols; ++x) {
        for(auto y=0;y<roi_b.rows;++y) {
            coord.x = x; coord.y = y;
            auto pxBrightness = roi_b.at<uchar>(coord);
            if(pxBrightness > mu_b[0]) {
                mb += pxBrightness;
                ++validPx;
            }
        }
    }

    mc /= (validPx > 0 ? validPx : 1);
    return cv::Scalar(ma,mb,mc);
}

bool VarianceClassifier::classify(cv::Mat1b &window) {
    cv::Scalar mu_d, sigma_d, mu_e, sigma_e;
    cv::meanStdDev(window(_D), mu_d, sigma_d);
    cv::meanStdDev(window(_E), mu_e, sigma_e);

    if(std::pow(sigma_d[0],2) < _t || std::pow(sigma_e[0],2) < _t) {
        return false;
    }

    cv::Scalar helper = _getMForABC(window);
    double ma = helper[0], mb = helper[1], mc = helper[2];
    if(mb < _k*ma || mb < _k*mc) {
        return false;
    }
    return true;

}

// Adjust the thresholds untile the face is marked as a valid face
// we suppose that face has the same dimension of _positiveMFI / _negativeMFI
void VarianceClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    QDirIterator *it = new QDirIterator(positiveTrainingSet);
    std::vector<double> positiveT, negativeT, positiveK, negativeK;
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

    it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);

        cv::Scalar mu_d, sigma_d, mu_e, sigma_e;
        cv::meanStdDev(face(_D), mu_d, sigma_d);
        cv::meanStdDev(face(_E), mu_e, sigma_e);

        positiveT.push_back(std::pow(sigma_d[0],2));
        positiveT.push_back(std::pow(sigma_e[0],2));
    }

    it = new QDirIterator(negativeTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat raw = cv::imread(fileName.toStdString());
        cv::Mat1b face = Preprocessor::gray(raw);
        face = Preprocessor::equalize(face);

        cv::Scalar mu_d, sigma_d, mu_e, sigma_e;
        cv::meanStdDev(face(_D), mu_d, sigma_d);
        cv::meanStdDev(face(_E), mu_e, sigma_e);

        negativeT.push_back(std::pow(sigma_d[0],2));
        negativeT.push_back(std::pow(sigma_e[0],2));
    }

    _t = equal_error_rate(positiveT,negativeT).second/2.1;
    std::cout << "T: << " << _t << "\nK: " << _k << std::endl;

    delete it;
}

/*
    cv::rectangle(face,_A,cv::Scalar(255,0,0));
    cv::rectangle(face,_B,cv::Scalar(255,255,0));
    cv::rectangle(face,_C,cv::Scalar(255,0,255));
    cv::rectangle(face,_D,cv::Scalar(0,0,0));
    cv::rectangle(face,_E,cv::Scalar(0,255,255));
*/
