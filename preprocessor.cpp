#include "preprocessor.h"

cv::Mat1b Preprocessor::gray(const cv::Mat &image) {
    cv::Mat gray;
    cv::Mat1b gray_norm;
    cv::cvtColor(image, gray_norm, CV_BGR2GRAY);
    //cv::normalize(gray_norm, gray_norm, 0, 255, CV_MINMAX);
    return gray_norm;
}

cv::Mat1b Preprocessor::equalize(const cv::Mat1b &gray) {
    cv::Mat1b equalizedImage;
    cv::equalizeHist(gray, equalizedImage);
    return equalizedImage;
}

cv::Mat1b Preprocessor::threshold(const cv::Mat1b &grad) {
    cv::Scalar mu, sigma;
    cv::meanStdDev(grad, mu, sigma);
    const double c = 0.89;
    // Scalar is a vector of quartets, we're working on grayscale thus we extract only the first channel
    double threshold = mu[0] + c * sigma[0];
    cv::Mat1b thresRes;
    cv::threshold(grad, thresRes, threshold, 255, CV_THRESH_BINARY);
    return thresRes;
}

cv::Mat1b Preprocessor::edge(const cv::Mat &image) {
    // now we can use the sobel operator to extract the edges of the equalized image
    // sobel operator calculate an approximation of the partial derivates, in order to find out
    // the light changing in an pixel neighborhood
    cv::Mat grad, data;
    // from the equalized image, save into grad the derivate along the x axes (order 1) (order 0 to y axes)
    // uses depth of 16 bit signed, to avoid overflow (the derivate can be less then zero)
    image.convertTo(data, CV_32FC1);

    cv::Sobel(data,grad,CV_32FC1,0,1);
    cv::Mat abs = cv::abs(grad);
    cv::normalize(abs, abs, 0, 255, CV_MINMAX);
    cv::Mat1b ret;
    abs.convertTo(ret, CV_8UC1);
    return ret;
}

cv::Mat1b Preprocessor::process(const cv::Mat &image) {
    // lets use the histogram equalization method in order to
    // equalize the distribution of greys in the original image
    // Thus we stretch the historgram trying to make it plan

    // first, convert the image to grayscale if is not in grayscale already
    cv::Mat1b grayImg = Preprocessor::gray(image);

    // second, equalize it
    cv::Mat1b equalizedImage = equalize(grayImg);

    // third, the edge detection
    cv::Mat1b grad = Preprocessor::edge(equalizedImage);

    // Now we apply a segmentation algorithm, in order to remove noise from the grayscale equalized edge detected image

    // Thresholding
    cv::Mat1b thresRes = threshold(grad);

    // last step of preprocessing, dilatation
    cv::Mat1b dilatationRes;
    cv::dilate(thresRes, dilatationRes, cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3), cv::Point(1,1)), cv::Point(1,1));
    return dilatationRes;
}

bool Preprocessor::validMime(QString fileName, QString _mimeFilter) {
    QMimeDatabase mimeDB;
    return mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
}
