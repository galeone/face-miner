#ifndef FACEFINDER_H
#define FACEFINDER_H

#include "faceclassifier.h"

class FaceFinder : public QObject
{
    Q_OBJECT

private:
    FaceClassifier *_fc;
public:
    std::vector<std::pair<cv::Rect, cv::Mat1b>> find(const cv::Mat &);

signals:
    void found(std::vector<std::pair<cv::Rect, cv::Mat1b>>);
    void ready();

public slots:
    void setClassifier(FaceClassifier *fc);
};

#endif // FACEFINDER_H
