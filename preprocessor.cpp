#include "preprocessor.h"

cv::Mat1b Preprocessor::edge(cv::Mat image) {
    // lets use the histogram equalization method in order to
    // equalize the distribution of greys in the original image
    // Thus we stretch the historgram trying to make it plan

    // first, convert the image to grayscale if is not in grayscale already
    if(image.channels() > 1) {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }

    // second, equalize it
    cv::Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);

    // now we can use the sobel operator to extract the edges of the equalized image
    // sobel operator calculate an approximation of the partial derivates, in order to find out
    // the light changing in an pixel neighborhood

    cv::Mat grad;
    // from the equalized image, save into grad the derivate along the x axes (order 1) (order 0 to y axes)
    // uses depth of 16 bit signed, to avoid overflow (the derivate can be less then zero)
    cv::Sobel(equalizedImage, grad, CV_16S, 1, 0);

    // converting from 16S to 8U (255 shades of gray LOL)
    cv::Mat1b grad_abs;
    cv::convertScaleAbs(grad, grad_abs);
    return grad_abs;
}

cv::Mat1b Preprocessor::process(cv::Mat image) {
    cv::Mat1b grad = Preprocessor::edge(image);
    // Now we apply a segmentation algorithm, in order to remove noise from the grayscale equalized edge detected image
    cv::Scalar mu, sigma;
    cv::meanStdDev(grad, mu, sigma);

    // Thresholding
    const double c = 1;
    // Scalar is a vector of quartets, we're working on grayscale thus we extract only the first channel
    double threshold = mu[0] + c * sigma[0];
    cv::Mat1b thresRes;
    cv::threshold(grad, thresRes, threshold, 255, CV_THRESH_BINARY);

    // last step of preprocessing, dilatation
    int structuringMatrix[3][3] = {
        {0,1,0},
        {1,1,1},
        {0,1,0}
    };
    cv::Mat structuringElement(3, 3, CV_8UC1, &structuringMatrix);
    cv::Mat1b dilatationRes;
    cv::dilate(thresRes, dilatationRes, structuringElement);
    return dilatationRes;
}

