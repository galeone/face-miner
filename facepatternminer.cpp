#include "facepatternminer.h"


#undef DEBUG


FacePatternMiner::FacePatternMiner(QString dataset, QString mimeFilter) {
    _dataset = dataset;
    _mimeFilter = mimeFilter;
    _edgeDir = new QDir("edge");
    if(!_edgeDir->exists()) {
        QDir().mkdir(_edgeDir->path());
    }

    auto setPath = _edgeDir->absolutePath().append(QString("/"));
    _positiveDB = new QFile(setPath + QString("positive.db"));
    _negativeDB = new QFile(setPath + QString("negative.db"));
    _imageSizeFile = new QFile(setPath + QString("image.size"));

#ifdef DEBUG
    // do not use previous computed db
    if(_positiveDB->exists()) {
        _positiveDB->remove();
    }
    _positiveDB->open(QFileDevice::WriteOnly);

    if(_negativeDB->exists()) {
        _negativeDB->remove();
    }
    _negativeDB->open(QFileDevice::WriteOnly);

    if(_imageSizeFile->exists()) {
        _imageSizeFile->remove();
        _imageSize = NULL;
    }
    _imageSizeFile->open(QFileDevice::WriteOnly);
#else
    // use previous computed db, if exists. This ReadWrite
    _positiveDB->open(QFileDevice::ReadWrite);
    _negativeDB->open(QFileDevice::ReadWrite);
    _imageSizeFile->open(QFileDevice::ReadWrite);

    if(_imageSizeFile->size() > 0) {
        // format: cols rows
        QTextStream in(_imageSizeFile);
        auto sizeLine = in.readLine();
        QStringList sizes = sizeLine.split(" ");
        if(sizes.count() != 2) {
            _imageSize = NULL;
        } else {
            _imageSize = new cv::Size(std::atoi(sizes[0].toStdString().c_str()), std::atoi(sizes[1].toStdString().c_str()));
        }
        _imageSizeFile->close();

    } else {
        _imageSize = NULL;
    }
#endif


}

inline bool FacePatternMiner::_validMime(QString fileName) {
    QMimeDatabase mimeDB;
    return  mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
}

// _appendToSet extracts the pixel position with value bin. It appends the extracted pattern to the database
// file using the MAFIA database syntax
void FacePatternMiner::_appendToSet(const cv::Mat1b &transaction, uchar bin, QFile *database) {
    if(!database->isOpen()) {
        throw new std::logic_error(database->fileName().toStdString()+" is not open");
    }
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
        // add newline
        line += "\n";
        database->write(line.c_str());
    }
}

void FacePatternMiner::_preprocess() {
    // If I need to fill the databases
    if(_positiveDB->size() == 0 && _negativeDB->size() == 0) {
        QDirIterator *it = new QDirIterator(_dataset);
        while(it->hasNext()) {
            auto fileName = it->next();
            if(!_validMime(fileName)) {
                continue;
            }

            auto image = cv::imread(fileName.toStdString());

            if(_imageSize == NULL) {
                _imageSize = new cv::Size(image.cols,image.rows);
                QTextStream out(_imageSizeFile);
                out << std::to_string(image.cols).c_str();
                out << " ";
                out << std::to_string(image.rows).c_str();
                _imageSizeFile->close();
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

            // additional, remove noise
            if(_imageSize->width > 100) {
                cv::medianBlur(equalizedImage,equalizedImage,5);
            }

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

            // saving .edge image, required by the classifier
            cv::imwrite(_edgeFileOf(fileName), grad);

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

            // Appending transaction (the image), to transaction database

            // Creating test set for positive pattern
            _appendToSet(dilatationRes, 255, _positiveDB);

            // Creating test set for negative pattern
            _appendToSet(dilatationRes, 0, _negativeDB);

        }
    }

    _positiveDB->close();
    _negativeDB->close();
}

// _mineMFI mines the Most Frequent Itemset in the database.
// The database must be in ASCII format, according to the MAFIA Syntax.
// Returns the MFI as a b/w image.
// minSupport is a parameter passed to MAFIA algoritm, to prune results in the depth first search
cv::Mat1b FacePatternMiner::_mineMFI(QFile *database, float minSupport, std::vector<cv::Point> &coordinates) {
    std::ostringstream out;
    out << std::setprecision(3) << minSupport;
    std::string minSupportStr(out.str());
    std::string ext(minSupportStr + ".mfi");

    QString mfiPath = database->fileName() + QString(ext.c_str());
    QFile mfiFile(mfiPath);
#ifndef DEBUG
    if(!mfiFile.exists()) {
#endif
        QString executing("./MAFIA -mfi ");
        executing.append(minSupportStr.c_str());
        executing.append(" -ascii ");
        executing.append(database->fileName());
        executing.append(" ");
        executing.append(mfiPath);
        std::cout << executing.toStdString() << std::endl;
        QProcess::execute(executing);
#ifndef DEBUG
    }
#endif

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
                coordinates.push_back(pos);
            }
        } else {
            break;
        }

        ++lineCount;
    }

    mfiFile.close();

    return ret;
}

std::string FacePatternMiner::_edgeFileOf(QString rawFile) {
    return _edgeDir->absolutePath().append(QString("/")).append(rawFile.split(QString("/")).last()).toStdString();
}


void FacePatternMiner::_buildClassifier() {

    // TODO: threshold learning process
    // Create a classifier.
    // Put every positive image into the classifier and adjust threshold until every image
    // is classified correctly
    // Do the same for the negative ones
    // The target is the have 100% true positive and 100% true negative
    // that's hard due to the noise. Thus we have to tradeoff and accept some false positive
    // and some false negative.

    // TODO: variance classifier (really required?)

    // (face) feature classifier:
    FeatureClassifier fc(_positiveMFICoordinates, _negativeMFICoordinates);

    // TODO: train with dataset of negative patterns(non face)
    // scale according to the dimension of the lowest item to match
    // void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )

    QDirIterator *it = new QDirIterator(_dataset);
    while(it->hasNext()) {
        auto fileName = it->next();
        if(!_validMime(fileName)) {
            continue;
        }

        auto raw = cv::imread(fileName.toStdString());
        auto edge = cv::imread(_edgeFileOf(fileName));

        fc.setData(raw, edge);
        float r1diff, r2diff;
        int age = 1;
        while(!fc.rule1(r1diff) || !fc.rule2(r2diff)) {
            std::cout << "AGE "<< age << std::endl;
            while(!fc.rule1(r1diff)) {
                fc.setT1(fc.getT1()+1);
                std::cout << "T1 = " << fc.getT1() << std::endl;
            }
            while(!fc.rule2(r2diff)) {
                fc.setT2(fc.getT2()+1);
                std::cout << "T2 = " << fc.getT2() << std::endl;
            }
        }
        std::cout << "Thresholds: " << fc.getT1() << " " << fc.getT2() << std::endl;
        std::cout << "Age: " << age << std::endl;
    }
}


// slot
void FacePatternMiner::start() {
    _preprocess();
    emit preprocessing_terminated();

    // TODO: scalare in qualche modo
    float positiveMinSupport = 0.67;
    _positiveMFI = _mineMFI(_positiveDB, positiveMinSupport, _positiveMFICoordinates);
    // minSupp for negative mfi is the highest value possible (1) because we need to speed up the computation
    _negativeMFI = _mineMFI(_negativeDB, 1, _negativeMFICoordinates);
    emit mining_terminated(_positiveMFI, _negativeMFI);
    _buildClassifier();
}
