#ifndef FACEPATTERNMINER_H
#define FACEPATTERNMINER_H

#include <QDirIterator>
#include <QMimeDatabase>
#include <QStringList>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "MAFIA/Mafia.h"

#define DEBUG

class FacePatternMiner : public QObject
{
    Q_OBJECT

private:
    QDirIterator *_it;
    QString _mimeFilter;
    QDir *_edgeDir;

    void _preprocess();
    void _mineMFI();
    inline bool _validMime(QString);
    void _appendToTestSet(const cv::Mat &);

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
