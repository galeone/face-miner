#include "facepatternminer.h"
//#undef DEBUG

FacePatternMiner::FacePatternMiner(QString positiveTestSet, QString negativeTestSet, QString mimeFilter) {
    _positiveTestSet = positiveTestSet;
    _negativeTestSet = negativeTestSet;
    _mimeFilter = mimeFilter;
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

inline bool FacePatternMiner::_validMime(QString fileName) {
    QMimeDatabase mimeDB;
    return  mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
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
        QDirIterator *it = new QDirIterator(_positiveTestSet);
        size_t count = 0;
        while(it->hasNext()) {
            auto fileName = it->next();
            if(!_validMime(fileName)) {
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
            // mined facial features are black, thus bin = 0
            _addTransactionToDB(res, 255,  _positiveDB);

            // Creating test set for negative pattern
            // bin = 255
            _addTransactionToDB(res, 0, _negativeDB);
            if(++count == 100) {
                break;
                delete it;
            }
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
    float positiveMinSupport = 0.98, negativeMinSupport = 0.5;
    _positiveMFI = _mineMFI(_positiveDB, positiveMinSupport, _positiveMFICoordinates);
    // minSupp for negative mfi is the highest value possible (1) because we need to speed up the computation
    _negativeMFI = _mineMFI(_negativeDB, negativeMinSupport, _negativeMFICoordinates);
    emit mining_terminated(_positiveMFI, _negativeMFI);
    _trainClassifiers();
    emit training_terminated();
    // Test, pick a random image.
    /*cv::Mat test = cv::imread("./datasets/BioID-FaceDatabase-V1.2/BioID_0921.pgm");
    _faceClassifier->classify(test);
    cv::namedWindow("test");
    cv::imshow("test", test); */
}

void FacePatternMiner::_trainClassifiers() {
    // Classifiers
    _varianceClassifier = new VarianceClassifier(_positiveMFI, _negativeMFI);
    _featureClassifier = new FeatureClassifier(_positiveMFICoordinates, _negativeMFICoordinates);
    _svmClassifier = new SVMClassifier();

    QDirIterator *it = new QDirIterator(_positiveTestSet);
    uint32_t totfile = 0;
    while(it->hasNext()) {
        ++totfile;
        auto fileName = it->next();
        if(!_validMime(fileName)) {
            continue;
        }
        cv::Mat faceGeneric = cv::imread(fileName.toStdString());
        cv::Mat1b faceGray;
        if(faceGeneric.channels() > 1) {
            cv::cvtColor(faceGeneric, faceGray, CV_BGR2GRAY);
        } else {
            faceGray = faceGeneric;
        }
        _varianceClassifier->train(faceGray);
        _featureClassifier->train(faceGray);
        emit preprocessing(faceGray);
    }
    delete it;
    std::cout << "[+] Variance classifier sucessully trained" << std::endl;

    // TODO: train feature classifier
    // TODO: create svm classifyer

    _faceClassifier = new FaceClassifier(_varianceClassifier,_featureClassifier,_svmClassifier, *_trainImageSize);
}


/*void FacePatternMiner::_buildClassifier() {

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

    QDirIterator *positiveIt = new QDirIterator(_positiveTestSet);
    QDirIterator *negativeIt = new QDirIterator(_negativeTestSet);
    while(positiveIt->hasNext()) {
        auto fileName = positiveIt->next();
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
        std::cout << "[P]Thresholds: " << fc.getT1() << " " << fc.getT2() << std::endl;
        std::cout << "[P]Age: " << age << std::endl;
    }

    while(negativeIt->hasNext()) {
        auto fileName = negativeIt->next();
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
        std::cout << "[N]Thresholds: " << fc.getT1() << " " << fc.getT2() << std::endl;
        std::cout << "[N]Age: " << age << std::endl;
    }
}
*/
