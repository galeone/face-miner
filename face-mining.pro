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
    videostreamview.cpp \
    facepatternminer.cpp \
    cantor.cpp \
    featureclassifier.cpp \
    varianceclassifier.cpp \
    svmclassifier.cpp

HEADERS  += facialrecognition.h \
    cameracalibrationworker.h \
    cv2qt.h \
    qt2cv.h \
    camstream.h \
    videostreamview.h \
    facepatternminer.h \
    cantor.h \
    iclassifier.h \
    featureclassifier.h \
    varianceclassifier.h \
    svmclassifier.h

FORMS    += facialrecognition.ui

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/release/ -lMAFIA
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/debug/ -lMAFIA
#else:unix: LIBS += -L$$PWD/build-MAFIA-Desktop-Debug/ -lMAFIA

#INCLUDEPATH += $$PWD/MAFIA
#DEPENDPATH += $$PWD/MAFIA

# http://dragly.org/2013/11/05/copying-data-files-to-the-build-directory-when-working-with-qmake/
copydata.commands = $(COPY_DIR) $$PWD/datasets/ $$OUT_PWD

CONFIG(release, debug|release): mafiaexe.commands = $(COPY) $$PWD/build-MAFIA-Desktop-Debug/MAFIA $$OUT_PWD
CONFIG(debug, debug|release): mafiaexe.commands = $(COPY) $$PWD/build-MAFIA-Desktop-Release/MAFIA $$OUT_PWD
first.depends = $(first) copydata mafiaexe
export(first.depends)
export(mafiaexe.commands)
export(copydata.commands)
QMAKE_EXTRA_TARGETS += first copydata mafiaexe
