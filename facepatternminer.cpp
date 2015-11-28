#include "facepatternminer.h"

FacePatternMiner::FacePatternMiner(QString dataset) {
    _it = new QDirIterator(dataset, QStringList() << "*.pgm", QDir::Files);
}

// slot
void FacePatternMiner::start() {
    while(_it->hasNext()) {
        auto image = cv::imread(_it->next().toStdString());
        emit preprocessing(image);
        // lets user the histogram equalization method in order to
        // equalize the distributuo of greys in the original image
        // Thus we stretch the historgram trying to make it plan

        // first, convert the image to grayscale
        cv::cvtColor(image, image, CV_BGR2GRAY);

        // second, equalize it
        cv::Mat equalizedImage;
        cv::equalizeHist(image, equalizedImage);

        // TODO: emettere un altro segnale e mostrare immagine affiancata frutto del preproressing
        emit preprocessing(equalizedImage);

        // now we can use the sobel operator to extract the edges of the equalized image
        // sobel operator calculate an approximation of the partial derivates, in order to find out
        // the light changing in an pixel neighborhood

        cv::Mat grad_x, grad_y;
        // from the equalized image, save into grad_x the derivate along the x axes (order 1) (order 0 to y axes)
        // uses depth of 16 bit signed, to avoid overflow (the derivate can be less then zero)
        cv::Sobel(equalizedImage, grad_x, CV_16S, 1, 0);
        // the same of above, but along the y axes
        cv::Sobel(equalizedImage, grad_y, CV_16S, 0, 1);

        // converting from 16S to 8U (255 shades of gray LOL)
        cv::Mat grad_x_abs, grad_y_abs;
        cv::convertScaleAbs(grad_x, grad_x_abs);
        cv::convertScaleAbs(grad_y, grad_y_abs);

        // try to approximate the gradient, using a weighted sum of the calculated partial derivates
        // The approxiamation is G = |G_x| + |G_y|
        cv::Mat grad;
        cv::addWeighted(grad_x_abs, 1, grad_y_abs, 1, 0, grad);

        emit preprocessing(grad);

        // Now we apply a segmentation algorithm, in order to remove noise from the grayscale equalized edge detected image
        cv::Scalar mu, sigma;

        cv::meanStdDev(grad,mu,sigma);

        // Thresholding
        const double c = 1;
        // Scalar is a vector of quartets, we're working on grayscale thus we extract only the first channel
        double threshold = mu[0] + c * sigma[0];
        std::cout << cv::mean(mu) << " " <<  cv::mean(sigma) << std::endl;
        cv::Mat thresRes;
        cv::threshold(grad,thresRes, threshold, 255, CV_THRESH_BINARY);

        emit preprocessing(thresRes);

        // last step of preprocessing, dilatation
        int structuringMatrix[3][3] = {
            {0,1,0},
            {1,1,1},
            {0,1,0}
        };
        cv::Mat structuringElement(3,3, CV_8UC1, &structuringMatrix);
        cv::Mat dilatationRes;
        cv::dilate(thresRes,dilatationRes,structuringElement);

        emit preprocessing(dilatationRes);
    }
}
