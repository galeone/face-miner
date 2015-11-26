#-------------------------------------------------
#
# Project created by QtCreator 2015-11-07T16:03:02
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceRecognition
TEMPLATE = app

CONFIG += c++14

LIBS += `pkg-config opencv --libs`

SOURCES += main.cpp\
        facialrecognition.cpp \
    cameracalibrationworker.cpp \
    cv2qt.cpp \
    qt2cv.cpp \
    camstream.cpp \
    videostreamview.cpp

HEADERS  += facialrecognition.h \
    cameracalibrationworker.h \
    cv2qt.h \
    qt2cv.h \
    camstream.h \
    videostreamview.h

FORMS    += facialrecognition.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/release/ -lMAFIA
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/debug/ -lMAFIA
else:unix: LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/ -lMAFIA

INCLUDEPATH += $$PWD/MAFIA
DEPENDPATH += $$PWD/MAFIA
