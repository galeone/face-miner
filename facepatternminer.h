#ifndef FACEPATTERNMINER_H
#define FACEPATTERNMINER_H

#include <QDirIterator>
#include <QStringList>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class FacePatternMiner : public QObject
{
    Q_OBJECT

private:
    QDirIterator *_it;

public:
    FacePatternMiner(QString dataset);

signals:
    void preprocessing(const cv::Mat &);
    void proprocessing_terminated();
    void mining_pattern(const cv::Mat &);
    void mining_terminated();

public slots:
   void  start();
};

#endif // FACEPATTERNMINER_H
