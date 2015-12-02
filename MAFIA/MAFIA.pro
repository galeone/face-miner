QT       -= core gui

TARGET = MAFIA
#TEMPLATE = lib

CONFIG += c++14

SOURCES += \
    Transaction.cpp \
    Mafia.cpp \
    BaseBitmap.cpp \
    Bitmap.cpp \
    ItemsetOutput.cpp

HEADERS += \
    Tables.h \
    ItemsetOutput.h \
    BaseBitmap.h \
    Bitmap.h \
    Transaction.h \
    TreeNode.h
