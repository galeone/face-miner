#include "svmclassifier.h"

using namespace cv::ml;

// rows1/2 are the positions in the window, where to ectract px intensities and haar features
SVMClassifier::SVMClassifier(const cv::Rect &rows1, const cv::Rect &rows2) {
    _r1 = rows1;
    _r2 = rows2;
    _svm = SVM::create();
    _featureVectorCard = _r1.width * (_r1.height + _r2.height);
    std::cout << _r1 << " " << _r2 << std::endl;
    std::cout << _featureVectorCard << " < size of feature vector" << std::endl;
}

//--------------------------------
// Wavelet transform
//--------------------------------
void SVMClassifier::_haarWavelet(cv::Mat src, cv::Mat &dst, int NIter) {
    float c,dh,dv,dd;
    assert( src.type() == CV_32FC1 );
    int width = src.cols;
    int height = src.rows;
    dst = cv::Mat1f(src.rows, src.cols,  CV_32FC1);

    for (int k=0;k<NIter;k++)
    {
        for (int y=0;y<(height>>(k+1));y++)
        {
            for (int x=0; x<(width>>(k+1));x++)
            {
                c=(src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)+src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5;
                dst.at<float>(y,x)=c;

                dh=(src.at<float>(2*y,2*x)+src.at<float>(2*y+1,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x+1))*0.5;
                dst.at<float>(y,x+(width>>(k+1)))=dh;

                dv=(src.at<float>(2*y,2*x)+src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)-src.at<float>(2*y+1,2*x+1))*0.5;
                dst.at<float>(y+(height>>(k+1)),x)=dv;

                dd=(src.at<float>(2*y,2*x)-src.at<float>(2*y,2*x+1)-src.at<float>(2*y+1,2*x)+src.at<float>(2*y+1,2*x+1))*0.5;
                dst.at<float>(y+(height>>(k+1)),x+(width>>(k+1)))=dd;
            }
        }
        dst.copyTo(src);
    }
}

void haar_2d ( int m, int n, double u[] )

//****************************************************************************80
//
//  Purpose:
//
//    HAAR_2D computes the Haar transform of an array.
//
//  Discussion:
//
//    For the classical Haar transform, M and N should be a power of 2.
//    However, this is not required here.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 March 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int M, N, the dimensions of the array.
//
//    Input/output, double U[M*N], the array to be transformed.
//
{
    int i;
    int j;
    int k;
    double s;
    double *v;

    s = sqrt ( 2.0 );

    v = new double[m*n];

    for ( j = 0; j < n; j++ )
    {
        for ( i = 0; i < m; i++ )
        {
            v[i+j*m] = u[i+j*m];
        }
    }
    //
    //  Determine K, the largest power of 2 such that K <= M.
    //
    k = 1;
    while ( k * 2 <= m )
    {
        k = k * 2;
    }
    //
    //  Transform all columns.
    //
    while ( 1 < k )
    {
        k = k / 2;

        for ( j = 0; j < n; j++ )
        {
            for ( i = 0; i < k; i++ )
            {
                v[i  +j*m] = ( u[2*i+j*m] + u[2*i+1+j*m] ) / s;
                v[k+i+j*m] = ( u[2*i+j*m] - u[2*i+1+j*m] ) / s;
            }
        }
        for ( j = 0; j < n; j++ )
        {
            for ( i = 0; i < 2 * k; i++ )
            {
                u[i+j*m] = v[i+j*m];
            }
        }
    }
    //
    //  Determine K, the largest power of 2 such that K <= N.
    //
    k = 1;
    while ( k * 2 <= n )
    {
        k = k * 2;
    }
    //
    //  Transform all rows.
    //
    while ( 1 < k )
    {
        k = k / 2;

        for ( j = 0; j < k; j++ )
        {
            for ( i = 0; i < m; i++ )
            {
                v[i+(  j)*m] = ( u[i+2*j*m] + u[i+(2*j+1)*m] ) / s;
                v[i+(k+j)*m] = ( u[i+2*j*m] - u[i+(2*j+1)*m] ) / s;
            }
        }

        for ( j = 0; j < 2 * k; j++ )
        {
            for ( i = 0; i < m; i++ )
            {
                u[i+j*m] = v[i+j*m];
            }
        }
    }
    delete [] v;
    return;
}
//****************************************************************************

// _getFeatures extract every feature required in the classification
// thus intensities + haar like features
void SVMClassifier::_getFeatures(const cv::Mat1b &window, cv::Mat1f &coeff) {
    coeff = cv::Mat1f(1, _featureVectorCard, CV_32FC1);

    cv::Mat1b face = Preprocessor::equalize(window);
    cv::Mat1b roi1 = face(_r1), roi2 = face(_r2);

    auto counter = 0;
    cv::Point pos(0,0);
/*
    for(auto row = 0; row < _r1.height; ++row) {
        pos.y = row;
        for(auto col=0;col<_r1.width;++col) {
            pos.x = col;
            coeff.at<float>(0, counter) = roi1.at<uchar>(pos);
            ++counter;
        }
    }

    for(auto row = 0; row < _r2.height; ++row) {
        pos.y = row;
        for(auto col=0;col<_r2.width;++col) {
            pos.x = col;
            coeff.at<float>(0, counter) = roi2.at<uchar>(pos);
            ++counter;
        }
    }
*/
    cv::Mat1f haar;
    cv::Mat1f roi1F, roi2F;
    roi1.convertTo(roi1F, CV_32FC1);
    roi2.convertTo(roi2F, CV_32FC1);

    //_haarWavelet(roi1F, haar, 2);
    int m = roi1F.cols, n = roi1F.rows;
    double u[m*n];
    auto count = 0;
    for(auto y=0;y<n;++y) {
        for(auto x=0;x<m;++x){
            u[count++] = roi1F.at<float>(y,x);
        }
    }

    haar_2d(m,n,u);
    count = 0;
    for(auto y=0;y<n;++y) {
        for(auto x=0;x<m;++x){
            coeff.at<float>(0, counter) = u[count];
            ++counter;
            ++count;
        }
    }

    m = roi2F.cols, n = roi2F.rows;
    double v[m*n];
    count = 0;
    for(auto y=0;y<n;++y) {
        for(auto x=0;x<m;++x){
            v[count++] = roi2F.at<float>(y,x);
        }
    }

    haar_2d(m,n,v);
    count = 0;
    for(auto y=0;y<n;++y) {
        for(auto x=0;x<m;++x){
            coeff.at<float>(0, counter) = v[count];
            ++counter;
            ++count;
        }
    }
}

// source must be CV1FC1
void SVMClassifier::_insertLineAtPosition(const cv::Mat1f &source, cv::Mat1f &dest, uint32_t position) {
    for(auto col = 0; col < source.cols; ++col) {
        dest.at<float>(position, col) = source.at<float>(0, col);
    }
}

bool SVMClassifier::classify(const cv::Mat1b &window) {
    cv::Mat1f coeff;
    _getFeatures(window, coeff);
    return _svm->predict(coeff) > 0;
}

void SVMClassifier::train(std::vector<cv::Mat1b> &truePositive, std::vector<cv::Mat1b> &falsePositive){
    const char *filename = "svm-trained.xml";
    /*
    _svm->load(filename);
    if(_svm->get_support_vector_count() > 0) { // trained model exist
        std::cout << "Using existing trained model" << std::endl;
        return;
    }
    */

    auto positiveCount = truePositive.size(), negativeCount = falsePositive.size();
    std::cout << "Positive n: " << positiveCount << "\nNegative n: " << negativeCount << std::endl;

    cv::Mat1i labels(positiveCount + negativeCount, 1, CV_32FC1);
    cv::Mat1f samples(positiveCount + negativeCount, _featureVectorCard, CV_32FC1);

    auto counter = 0;
    for(const auto &face : truePositive) {
        labels.at<int>(counter, 0) = 1;

        cv::Mat1f row;
        _getFeatures(face, row);
        _insertLineAtPosition(row, samples, counter);

        ++counter;
    }

    for(const auto &face : falsePositive) {
        labels.at<int>(counter, 0) = -1;

        cv::Mat1f row;
        _getFeatures(face, row);
        _insertLineAtPosition(row, samples, counter);

        ++counter;
    }

    // Set up SVM's parameters
    _svm->setType(SVM::C_SVC);
    _svm->setKernel(SVM::RBF);
    _svm->setC(2.5);
    _svm->setGamma(1e-5);

    cv::Ptr<TrainData> tData = TrainData::create(samples,SampleTypes::ROW_SAMPLE,labels);
    _svm->train(tData);
    _svm->save(filename);

    std::cout << "[!] SVM trained successfully" << std::endl;
}

void SVMClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    std::vector<cv::Mat1b> positive, negative;

    QDirIterator *it = new QDirIterator(positiveTrainingSet);
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
