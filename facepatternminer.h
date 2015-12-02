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
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cantor.h"

#define DEBUG

class FacePatternMiner : public QObject
{
    Q_OBJECT

private:
    QDirIterator *_it;
    QString _mimeFilter;
    QDir *_edgeDir;
    QFile *_positiveDB, *_negativeDB;
    cv::Size *_imageSize;

    void _preprocess();
    cv::Mat1b _mineMFI(QFile *,float);
    inline bool _validMime(QString);
    void _appendToSet(const cv::Mat1b &, uchar , QFile*);

public:
    FacePatternMiner(QString dataset, QString mineFilter);

signals:
    void preprocessing(const cv::Mat &);
    void proprocessing_terminated();
    void mining_pattern(const cv::Mat &);
    void mining_terminated();

public slots:
   void  start();
};

#endif // FACEPATTERNMINER_H
