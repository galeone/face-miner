#ifndef VideoStreamView_H
#define VideoStreamView_H

#include <QLabel>
#include <Qt>
#include <QKeyEvent>
#include <QMouseEvent>
#include "qt2cv.h"
#include <iostream>

class VideoStreamView : public QLabel
{
    Q_OBJECT

public:
    explicit VideoStreamView(QWidget *parent = 0) : QLabel(parent) {}

signals:
    void clicked(const cv::Point &position);


protected:
    virtual void keyPressEvent(QKeyEvent* ev);
    virtual void mousePressEvent(QMouseEvent* ev);


};

#endif // VideoStreamView_H
