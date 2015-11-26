QT       -= core gui

TARGET = MAFIA
TEMPLATE = lib

CONFIG += c++14

SOURCES += \
    BaseBitmap.cpp \
    Bitmap.cpp \
    ItemsetOutput.cpp \
    Mafia.cpp \
    Transaction.cpp

HEADERS += \
    BaseBitmap.h \
    Bitmap.h \
    ItemsetOutput.h \
    Tables.h \
    Transaction.h \
    TreeNode.h
