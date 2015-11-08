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
    camstreamview.cpp \
    cv2qt.cpp \
    qt2cv.cpp \
    camstream.cpp

HEADERS  += facialrecognition.h \
    cameracalibrationworker.h \
    camstreamview.h \
    cv2qt.h \
    qt2cv.h \
    camstream.h

FORMS    += facialrecognition.ui
