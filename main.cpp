#include "facialrecognition.h"
#include <QApplication>
#include "BaseBitmap.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    FacialRecognition w;
    w.show();

    return a.exec();
}
