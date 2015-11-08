#ifndef CAMSTREAMVIEW_H
#define CAMSTREAMVIEW_H

#include <QLabel>
#include <QKeyEvent>
#include <QMouseEvent>
#include "qt2cv.h"
#include <iostream>

class CamStreamView : public QLabel
{
    Q_OBJECT

public:
    explicit CamStreamView(QWidget *parent = 0) : QLabel(parent) {}

signals:
    void clicked(const cv::Point &position);

protected:
    virtual void keyPressEvent(QKeyEvent* ev);
    virtual void mousePressEvent(QMouseEvent* ev);


};

#endif // CAMSTREAMVIEW_H
