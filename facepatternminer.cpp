#include "facepatternminer.h"
FacePatternMiner::FacePatternMiner(QString dataset, QString mimeFilter) {
    _mimeFilter = mimeFilter;
    _it = new QDirIterator(dataset);
    _edgeDir = new QDir("edge");
    if(!_edgeDir->exists()) {
        QDir().mkdir(_edgeDir->path());
    }
    auto setPath = _edgeDir->absolutePath().append(QString("/"));
    _positiveDB = new QFile(setPath + QString("positive.db"));
    if(_positiveDB->exists()) {
        _positiveDB->remove();
    }
    _negativeDB = new QFile(setPath + QString("negative.db"));
    if(_negativeDB->exists()) {
        _negativeDB->remove();
    }

    _imageSize = NULL;
}

inline bool FacePatternMiner::_validMime(QString fileName) {
    QMimeDatabase mimeDB;
    return  mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
}

// _appendToSet extracts the pixel position with value bin. It appends the extracted pattern to the database
// file using the MAFIA database syntax
void FacePatternMiner::_appendToSet(const cv::Mat1b &transaction, uchar bin, QFile *database) {
    database->open(QFileDevice::Append);
    std::string line = "";
    for(auto x=0;x<transaction.cols;++x) {
        for(auto y=0;y<transaction.rows;++y) {
            cv::Point point(x,y);
            if(transaction.at<uchar>(point) == bin) {
                // Use Cantor::pair to create a bijective association between the point and a number in the database
                line += std::to_string(Cantor::pair(point)) + " ";
            }
        }
    }
    if(line.length() > 0) {
        line.pop_back(); // remove last space after closing bracket
    }
    // add newline
    line += "\n";
    database->write(line.c_str());
    database->close();

}

// Costruire il training database (proprio un file che contiene le edge image estratte)
/*Datasets can be in ASCII or binary format. For ASCII files, the file format must be:
[item_id_1] [item_id_2] ... [item_id_n]
Items do not have to be sorted within each transaction. Items are separated by spaces and each transaction should end with a newline, e.g.
1 4 2
2 8 9 4
2 5
*/
// Dato che usa gli spazi come separatori di item e i \n come separatore di transaction, creiamo il dataset in questo formato
// (x,y) dove x,y sono le coordiante del px bianco nella i-esima immagine
void FacePatternMiner::_preprocess() {
    while(_it->hasNext()) {
        auto fileName = _it->next();
        if(!_validMime(fileName)) {
            continue;
        }

        auto image = cv::imread(fileName.toStdString());

        if(_imageSize == NULL) {
           _imageSize = new cv::Size(image.cols,image.rows);
        }

        emit preprocessing(image);
        // lets user the histogram equalization method in order to
        // equalize the distribution of greys in the original image
        // Thus we stretch the historgram trying to make it plan

        // first, convert the image to grayscale if is not in grayscale already
        if(image.channels() > 1) {
            cv::cvtColor(image, image, CV_BGR2GRAY);
        }

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
#ifdef DEBUG
        std::string edgeFile = _edgeDir->absolutePath().append(QString("/")).append(fileName.split(QString("/")).last()).toStdString();
        cv::imwrite(edgeFile,dilatationRes);
#endif
        // Appending transaction (the image), to transaction database

        // Creating test set for positive pattern
        _appendToSet(dilatationRes, 255, _positiveDB);

        // Creating test set for negative pattern
        _appendToSet(dilatationRes, 0, _negativeDB);
    }

}

// _mineMFI mines the Most Frequent Itemset in the database.
// The database must be in ASCII format, according to the MAFIA Syntax.
// Returns the MFI as a b/w image.
// minSupport is a parameter passed to MAFIA algoritm, to prune results in the depth first search
cv::Mat1b FacePatternMiner::_mineMFI(QFile *database, float minSupport) {
    std::ostringstream out;
    out << std::setprecision(3) << minSupport;
    std::string minSupportStr(out.str());
    std::string ext(minSupportStr + ".mfi");

    QString mfiPath = database->fileName() + QString(ext.c_str());
    QFile mfiFile(mfiPath);
    if(!mfiFile.exists()) {
        QString executing("./MAFIA -mfi ");
        executing.append(minSupportStr.c_str());
        executing.append(" -ascii ");
        executing.append(database->fileName());
        executing.append(" ");
        executing.append(mfiPath);
        std::cout << executing.toStdString() << std::endl;
        QProcess::execute(executing);
    }

    cv::Mat1b ret = cv::Mat1b::zeros(*_imageSize);

    if(!mfiFile.open(QFileDevice::ReadOnly)) {
        throw new std::runtime_error("Unable to open " + mfiPath.toStdString());
    }

    QTextStream in(&mfiFile);

    auto lineCount = 1;

    while(!in.atEnd()) {
         QString line = in.readLine();
         QStringList coords = line.split(" ");
         if(coords.length() > 0) {
             // the last element of the line (minSupp is between brackets)
            coords.pop_back();
            for(const QString &coord : coords) {
                auto pos = Cantor::unpair(std::stoi(coord.toStdString()));
                ret.at<uchar>(pos) = 255;
            }
         } else {
             break;
         }

         ++lineCount;
    }

    mfiFile.close();

    return ret;
}

// slot
void FacePatternMiner::start() {
    _preprocess();
    emit proprocessing_terminated();
    auto positiveMFI = _mineMFI(_positiveDB, 0.66);
    cv::namedWindow("positive MFI");
    cv::imshow("positive MFI", positiveMFI);
    auto negativeMFI = _mineMFI(_negativeDB, 0.99);
    cv::namedWindow("negative MFI");
    cv::imshow("negative MFI", negativeMFI);
    emit mining_terminated();
}
