#include "videostreamview.h"

void VideoStreamView::keyPressEvent(QKeyEvent* ev)
{
    switch(ev->key()) {
    case Qt::Key::Key_Space:
        break;
    }
}

void VideoStreamView::mousePressEvent(QMouseEvent* ev)
{
    emit clicked(Qt2Cv::qpointToCvPoint(ev->pos()));
}
