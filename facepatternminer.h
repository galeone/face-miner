#ifndef FACEPATTERNMINER_H
#define FACEPATTERNMINER_H

#include <QDirIterator>
#include <QMimeDatabase>
#include <QStringList>
#include <QTextStream>
#include <QStringList>
#include <QProcess>
#include <iostream>
#include <iomanip>
#include <vector>
#include <QSet>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cantor.h"
#include "featureclassifier.h"
#include "varianceclassifier.h"
#include "svmclassifier.h"
#include "featureclassifier.h"
#include "faceclassifier.h"

#define DEBUG

class FacePatternMiner : public QObject
{
    Q_OBJECT

private:
    QString _mimeFilter, _positiveTrainSet, _negativeTrainSet, _positiveTestSet, _negativeTestSet;
    QDir *_edgeDir;
    QFile *_positiveDB, *_negativeDB, *_trainImageSizeFile;
    cv::Size *_trainImageSize;
    cv::Mat1b _positiveMFI, _negativeMFI;
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
    VarianceClassifier *_varianceClassifier;
    FeatureClassifier *_featureClassifier;
    SVMClassifier *_svmClassifier;

    FaceClassifier *_faceClassifier;

    void _preprocess();
    void _trainClassifiers();

    inline bool _validMime(QString);
    void _addTransactionToDB(const cv::Mat1b &, uchar , QFile*);
    cv::Mat1b _mineMFI(QFile *,float, std::vector<cv::Point> &);

public:
    FacePatternMiner(QString train_positive, QString train_negative, QString test_positive, QString test_negative, QString mime = "image/x-portable-graymap");

signals:
    void preprocessing(const cv::Mat &);
    void preprocessed(const cv::Mat &);
    void preprocessing_terminated(); // starting mining
    void mining_terminated(const cv::Mat &positiveMFI, const cv::Mat &negativeMFI); // starting training
    void training_terminated();
    void built_classifier(FaceClassifier *classifier);

public slots:
   void  start();
};

#endif // FACEPATTERNMINER_H
