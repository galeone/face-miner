#include "facepatternminer.h"
//#undef DEBUG
FacePatternMiner::FacePatternMiner(QString train_positive, QString train_negative, QString test_positive, QString test_negative, QString mime) {
    _positiveTrainSet = train_positive;
    _negativeTrainSet = train_negative;

    _positiveTestSet = test_positive;
    _negativeTestSet = test_negative;

    _mimeFilter = mime;
    _edgeDir = new QDir("edge");
    if(!_edgeDir->exists()) {
        QDir().mkdir(_edgeDir->path());
    }

    auto setPath = _edgeDir->absolutePath().append(QString("/"));
    _positiveDB = new QFile(setPath + QString("positive.db"));
    _negativeDB = new QFile(setPath + QString("negative.db"));
    _trainImageSizeFile = new QFile(setPath + QString("image.size"));

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

    if(_trainImageSizeFile->exists()) {
        _trainImageSizeFile->remove();
    }
    _trainImageSize = NULL;
    _trainImageSizeFile->open(QFileDevice::WriteOnly);
#else
    // use previous computed db, if exists. This ReadWrite
    _positiveDB->open(QFileDevice::ReadWrite);
    _negativeDB->open(QFileDevice::ReadWrite);
    _trainImageSizeFile->open(QFileDevice::ReadWrite);

    if(_trainImageSizeFile->size() > 0) {
        // format: cols rows
        QTextStream in(_trainImageSizeFile);
        auto sizeLine = in.readLine();
        QStringList sizes = sizeLine.split(" ");
        if(sizes.count() != 2) {
            _trainImageSize = NULL;
        } else {
            _trainImageSize = new cv::Size(std::atoi(sizes[0].toStdString().c_str()), std::atoi(sizes[1].toStdString().c_str()));
        }
        _trainImageSizeFile->close();

    } else {
        _trainImageSize = NULL;
    }
#endif
}

// _appendToSet extracts the pixel position with value bin. It appends the extracted pattern to the database
// file using the MAFIA database syntax
void FacePatternMiner::_addTransactionToDB(const cv::Mat1b &transaction, uchar bin, QFile *database) {
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
        QDirIterator *it = new QDirIterator(_positiveTrainSet);
        //size_t count = 0;
        while(it->hasNext()) {
            auto fileName = it->next();
            if(!Preprocessor::validMime(fileName)) {
                continue;
            }

            auto image = cv::imread(fileName.toStdString());

            if(_trainImageSize == NULL) {
                _trainImageSize = new cv::Size(image.cols,image.rows);
                QTextStream out(_trainImageSizeFile);
                out << std::to_string(image.cols).c_str();
                out << " ";
                out << std::to_string(image.rows).c_str();
                _trainImageSizeFile->close();
            }

            emit preprocessing(image);
            std::cout << "before size: " << image.rows <<  " " << image.cols << std::endl;
            cv::Mat1b res = Preprocessor::process(image);
            std::cout << "after size: " << res.rows <<  " " << res.cols << std::endl;
            emit preprocessed(res);

            // Appending transaction (the image), to transaction database

            // Creating test set for positive pattern
            _addTransactionToDB(res, 255,  _positiveDB);

            // Creating test set for negative pattern
            _addTransactionToDB(res, 0, _negativeDB);

            /*if(++count == 100) {
                break;
                delete it;
            }
            */
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

    if(!mfiFile.open(QFileDevice::ReadOnly)) {
        throw new std::runtime_error("Unable to open " + mfiPath.toStdString());
    }

    QTextStream in(&mfiFile);
    auto lineCount = 1;
    QSet<int> coordNumSet;
    while(!in.atEnd()) {
        QString line = in.readLine();
        QStringList coords = line.split(" ");
        if(coords.length() > 0) {
            // the last element of the line (minSupp is between brackets)
            coords.pop_back();
            for(const QString &coord : coords) {
                coordNumSet.insert(std::stoi(coord.toStdString()));
            }
        } else {
            break;
        }
        ++lineCount;
    }

    mfiFile.close();

    cv::Mat1b ret = cv::Mat1b::zeros(*_trainImageSize);
    for(auto coordNum : coordNumSet){
        cv::Point pos = Cantor::unpair(coordNum);
        coordinates.push_back(pos);
        ret.at<uchar>(pos) = 255;
    }

    return ret;
}

// slot
void FacePatternMiner::start() {
    _preprocess();
    emit preprocessing_terminated();
    float positiveMinSupport = 0.92, negativeMinSupport = 0.9;
    _positiveMFI = _mineMFI(_positiveDB, positiveMinSupport, _positiveMFICoordinates);
    _negativeMFI = _mineMFI(_negativeDB, negativeMinSupport, _negativeMFICoordinates);

    emit mining_terminated(_positiveMFI, _negativeMFI);
    _trainClassifiers();

    // Test, pick a random image.
    //cv::Mat test = cv::imread("./datasets/mitcbcl/test/face/cmu_0000.pgm");
    cv::Mat test = cv::imread("./datasets/test.jpg");
    _faceClassifier->classify(test);
    cv::namedWindow("test1");
    cv::imshow("test1", test);
    cv::Mat test2 = cv::imread("./datasets/BioID-FaceDatabase-V1.2/BioID_0921.pgm");
    _faceClassifier->classify(test2);
    cv::namedWindow("test2");
    cv::imshow("test2", test2);
}

void FacePatternMiner::_trainClassifiers() {
    // Classifiers1
    _varianceClassifier = new VarianceClassifier(*_trainImageSize);
    _featureClassifier = new FeatureClassifier(_positiveMFICoordinates, _negativeMFICoordinates);
    _svmClassifier = new SVMClassifier();

    _varianceClassifier->train(_positiveTrainSet, _negativeTrainSet);
    _featureClassifier->train(_positiveTrainSet, _negativeTrainSet);

    std::cout << "[+] Classifiers sucessully trained" << std::endl;

    _faceClassifier = new FaceClassifier(_varianceClassifier,_featureClassifier,_svmClassifier, *_trainImageSize);
}
