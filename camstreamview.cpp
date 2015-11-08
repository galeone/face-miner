#include "camstreamview.h"

void CamStreamView::keyPressEvent(QKeyEvent* ev)
{
}

void CamStreamView::mousePressEvent(QMouseEvent* ev)
{
    emit clicked(Qt2Cv::qpointToCvPoint(ev->pos()));
}
