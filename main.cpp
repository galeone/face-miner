#include "facialrecognition.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    FacialRecognition w;
    w.show();

    return a.exec();
}
