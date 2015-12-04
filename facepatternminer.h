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
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cantor.h"
#include "featureclassifier.h"
#include "varianceclassifier.h"
#include "svmclassifier.h"
#include "featureclassifier.h"

#define DEBUG

class FacePatternMiner : public QObject
{
    Q_OBJECT

private:
    QString _mimeFilter, _positiveTestSet, _negativeTestSet;
    QDir *_edgeDir;
    QFile *_positiveDB, *_negativeDB, *_imageSizeFile;
    cv::Size *_imageSize;
    cv::Mat1b _positiveMFI, _negativeMFI;
    std::vector<cv::Point> _positiveMFICoordinates, _negativeMFICoordinates;
    VarianceClassifier *_varianceClassifier;

    inline bool _validMime(QString);
    void _preprocess();
    void _appendToSet(const cv::Mat1b &, uchar , QFile*);
    cv::Mat1b _mineMFI(QFile *,float, std::vector<cv::Point> &);
    std::string _edgeFileOf(QString);
    void _trainClassifiers();

public:
    FacePatternMiner(QString, QString, QString);

signals:
    void preprocessing(const cv::Mat &);
    void preprocessing_terminated();
    void mining_pattern(const cv::Mat &);
    void mining_terminated(const cv::Mat &, const cv::Mat &);

public slots:
   void  start();
};

#endif // FACEPATTERNMINER_H
