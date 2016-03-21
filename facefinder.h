/*
Face Miner: data mining applied to face detection
Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
Exhibit B is not attached; this software is compatible with the
licenses expressed under Section 1.12 of the MPL v2.
*/

#ifndef FACEFINDER_H
#define FACEFINDER_H

#include "faceclassifier.h"

class FaceFinder : public QObject
{
    Q_OBJECT

private:
    FaceClassifier *_fc;
public:
    std::vector<std::pair<cv::Rect, cv::Mat1b>> find(const cv::Mat &);

signals:
    void found(std::vector<std::pair<cv::Rect, cv::Mat1b>>);
    void ready();

public slots:
    void setClassifier(FaceClassifier *fc);
};

#endif // FACEFINDER_H
