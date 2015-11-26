#include "facialrecognition.h"
#include <QApplication>
#include "BaseBitmap.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    FacialRecognition w;
    auto x = new BaseBitmap(10);
    (void)x->_count;
    w.show();

    return a.exec();
}
