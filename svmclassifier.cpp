#include "svmclassifier.h"

// rows1/2 are the positions in the window, where to ectract px intensities and haar features
SVMClassifier::SVMClassifier(const cv::Rect &rows1, const cv::Rect &rows2, QString test_positive, QString test_negative) {
    _testPositive = test_positive;
    _testNegative = test_negative;
    _r1 = rows1;
    _r2 = rows2;
    _svm = new cv::SVM();
    _featureVectorCard = _r1.width * (_r1.height + _r2.height);
    std::cout << _r1 << " " << _r2 << std::endl;
    std::cout << _featureVectorCard << " < size of feature vector" << std::endl;
    //exit(-1);
    //_featureVectorCard = _r1.width * (_r1.height + _r2.height);
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
//****************************************************************************80

// _getFeatures extract every feature required in the classification
// thus intensities + haar like features
void SVMClassifier::_getFeatures(const cv::Mat1b &window, cv::Mat1f &coeff) {
    coeff = cv::Mat1f(1, _featureVectorCard, CV_32FC1);
    cv::Mat1b roi1 = window(_r1), roi2 = window(_r2);

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

    // intensities. Now coefficents of the haar transform
    // come faccio ad ottenere tante features quanti sono i px contenenti l'immagine se
    // per ottenere le features sono fare la differenze di rettangoli appartenenti all immagine
    // integrale facendo scorrere una window su questa? WTF.

    // al momento provo solo a testare sbattendo dentro al feature vector ogni punto dell'immagine integrale
    // se non va un cazzo cerco di capire come si fa con le haar like feature (anche se sul paper parla di coefficienti
    // della trasofrmata di haar. Che vabbé).

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

bool SVMClassifier::classify(cv::Mat1b &window) {
    cv::Mat1f coeff;
    _getFeatures(window, coeff);

    auto predictedLabel = _svm->predict(coeff);

    if(predictedLabel < 0 ) { // non face threshold
        return false;
    }
    return true;
}

void SVMClassifier::train(QString positiveTrainingSet, QString negativeTrainingSet) {
    std::cout << "[!] SVM classfier:\n";
    const char *filename = "svm-trained.xml";
    _svm->load(filename);
    if(_svm->get_support_vector_count() > 0) { // trained model exist
        std::cout << "Using existing trained model" << std::endl;
        Stats::print(_testPositive, _testNegative, this);
        return;
    }

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
            samples(positiveCount + negativeCount,_featureVectorCard, CV_32FC1);

    auto counter = 0;

    it = new QDirIterator(positiveTrainingSet);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!Preprocessor::validMime(fileName)) {
            continue;
        }

        cv::Mat face = cv::imread(fileName.toStdString());
        face = Preprocessor::gray(face);
        face = Preprocessor::equalize(face);

        labels.at<float>(counter, 0) = 1;

        cv::Mat1f row;
        _getFeatures(face, row);
        _insertLineAtPosition(row, samples, counter);

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
        face = Preprocessor::equalize(face);

        labels.at<float>(counter, 0) = -1;

        cv::Mat1f row;
        _getFeatures(face, row);
        _insertLineAtPosition(row, samples, counter);

        ++counter;
    }

    delete it;

    // Set up SVM's parameters
    CvSVMParams params;

    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    //params.gamma = 10;

    //params.term_crit   = cv::TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    //_svm->train(samples, labels, cv::Mat(), cv::Mat(),params);
    _svm->train_auto(samples,labels,cv::Mat(), cv::Mat(),params);
    _svm->save(filename);

    std::cout << "Trained successfull" << std::endl;
    Stats::print(_testPositive, _testNegative, this);

}
