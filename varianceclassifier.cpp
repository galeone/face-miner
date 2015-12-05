#include "varianceclassifier.h"

VarianceClassifier::VarianceClassifier(const cv::Mat &positiveMFI, const cv::Mat &negativeMFI) {
    _positiveMFI = positiveMFI;
    _negativeMFI = negativeMFI;

    auto cols = _positiveMFI.cols,
         rows = _positiveMFI.rows;

    auto aThirdRows = std::floor(rows/3),
         aThirdCols = std::floor(cols/3);

    // Mouth region
    _E = cv::Rect(0, rows-aThirdRows, cols, aThirdRows);

    // Nose region
    _D = cv::Rect(0, rows-2*aThirdRows, cols, aThirdRows);

    // Left eye region
    auto ac_cols = aThirdCols + 2;
    auto b_cols = cols - 2*ac_cols;

    // Top section
    if(rows % 2 != 0) {
        ++aThirdRows;
    }
    _A = cv::Rect(0, 0, ac_cols, aThirdRows);

    // Between eye region
    _B = cv::Rect(ac_cols, 0, b_cols, aThirdRows);

    // Right eye region
    _C = cv::Rect(cols - ac_cols, 0, ac_cols, aThirdRows);

    _t = 0;
    _k = 1;
    _trainingNumber = 0;
}

cv::Scalar VarianceClassifier::_getMForABC(cv::Mat &window) {
    cv::Scalar mu_a, mu_b, mu_c;
    cv::Mat1b roi_a = window(_A), roi_b = window(_B), roi_c = window(_C);
    mu_a = cv::mean(roi_a);
    mu_b = cv::mean(roi_b);
    mu_c = cv::mean(roi_c);

    float ma = 0, mb = 0, mc = 0;

    uint32_t validPx = 0;

    //ma is the average intensity of those pixels that are
    // darker than the averate intensity in region A
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

    ma /= validPx > 0 ? validPx : 1;
    validPx = 0;

    //mb is the average intensity of those pixels that are
    // birghter than the averate intensity in region B
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
    mb /= validPx > 0 ? validPx : 1;
    validPx = 0;

    //mc is the average intensity of those pixels that are
    // darker than the averate intensity in region C
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
    mc /= validPx > 0 ? validPx : 1;
    return cv::Scalar(ma,mb,mc);
}

bool VarianceClassifier::classify(cv::Mat1b &window) {
    cv::Scalar mu_d, sigma_d, mu_e, sigma_e;
    cv::meanStdDev(window(_D), mu_d, sigma_d);
    cv::meanStdDev(window(_E), mu_e, sigma_e);

    cv::Scalar helper = _getMForABC(window);
    double ma = helper[0], mb = helper[1], mc = helper[2];

    if(sigma_d[0] < _t && sigma_e[0] < _t)
        return false;

    if(mb < _k*ma || mb < _k*mc) {
        return false;
    }

    return true;
}

// Adjust the thresholds untile the face is marked as a valid face
// we suppose that face has the same dimension of _positiveMFI / _negativeMFI
void VarianceClassifier::train(cv::Mat1b &face) {
    cv::Scalar mu_d, sigma_d, mu_e, sigma_e;
    cv::meanStdDev(face(_D), mu_d, sigma_d);
    cv::meanStdDev(face(_E), mu_e, sigma_e);

   _oldT = _t;
    while(sigma_d[0] > _t || sigma_e[0] > _t) {
        _t += 1.27;
    }
    _t = (_t + _oldT) / 2;

    ++_trainingNumber;
    std::cout << "Item: " << _trainingNumber << "\nComputed threshold: " << _t << std::endl;

    cv::Scalar mu_a, mu_b, mu_c;
    cv::Mat1b roi_a = face(_A), roi_b = face(_B), roi_c = face(_C);
    mu_a = cv::mean(roi_a);
    mu_b = cv::mean(roi_b);
    mu_c = cv::mean(roi_c);

    cv::Scalar helper = _getMForABC(face);
    double ma = helper[0], mb = helper[1], mc = helper[2];

    while(mb < _k*ma || mb < _k*mc) {
        _k-=0.01;
    }

    std::cout << "Computed k: " << _k << std::endl;
/*
    cv::rectangle(face,_A,cv::Scalar(255,0,0));
    cv::rectangle(face,_B,cv::Scalar(255,255,0));
    cv::rectangle(face,_C,cv::Scalar(255,0,255));
    cv::rectangle(face,_D,cv::Scalar(0,0,0));
    cv::rectangle(face,_E,cv::Scalar(0,255,255));
*/
}
