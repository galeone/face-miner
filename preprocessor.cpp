#include "preprocessor.h"

cv::Mat1b Preprocessor::gray(cv::Mat &image) {
    cv::Mat1b gray;
    if(image.channels() > 1) {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    } else {
        gray = image;
    }
    return gray;
}

cv::Mat1b Preprocessor::equalize(cv::Mat1b &gray) {
    cv::Mat1b equalizedImage;
    cv::equalizeHist(gray, equalizedImage);
    return equalizedImage;
}

cv::Mat1b Preprocessor::threshold(cv::Mat1b &grad) {
    cv::Scalar mu, sigma;
    cv::meanStdDev(grad, mu, sigma);
    const double c = 1.15;
    // Scalar is a vector of quartets, we're working on grayscale thus we extract only the first channel
    double threshold = mu[0] + c * sigma[0];
    cv::Mat1b thresRes;
    cv::threshold(grad, thresRes, threshold, 255, CV_THRESH_BINARY);
    return thresRes;
}

cv::Mat1b Preprocessor::edge(cv::Mat &image) {
    // now we can use the sobel operator to extract the edges of the equalized image
    // sobel operator calculate an approximation of the partial derivates, in order to find out
    // the light changing in an pixel neighborhood
    cv::Mat grad;
    // from the equalized image, save into grad the derivate along the x axes (order 1) (order 0 to y axes)
    // uses depth of 16 bit signed, to avoid overflow (the derivate can be less then zero)

    /*cv::Mat ker = (cv::Mat_<char>(3,3)<<
                   1,2,1,
                   0,0,0,
                   -1,-2,-1
                   );

    cv::filter2D(equalizedImage,grad,CV_16S,ker);*/

    cv::Sobel(image,grad,CV_16S,0,1);

    // converting from 16S to 8U (255 shades of gray LOL)
    cv::Mat grad_abs;
    cv::convertScaleAbs(grad, grad_abs);

    return grad_abs;
}

cv::Mat1b Preprocessor::process(cv::Mat &image) {
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
    cv::Mat structuringElement = (cv::Mat_<uchar>(3,3)<<
                                  0,1,0,
                                  1,1,1,
                                  0,1,0);
    cv::Mat1b dilatationRes;
    //cv::dilate(thresRes, dilatationRes, structuringElement,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
    cv::dilate(thresRes,dilatationRes,structuringElement);
    return dilatationRes;
}

bool Preprocessor::validMime(QString fileName, QString _mimeFilter) {
    QMimeDatabase mimeDB;
    return mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
}
