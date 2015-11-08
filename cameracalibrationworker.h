#ifndef CAMERACALIBRATIONWORKER_H
#define CAMERACALIBRATIONWORKER_H

#include <QObject>

class CameraCalibrationWorker : public QObject
{
    Q_OBJECT

public:
    explicit CameraCalibrationWorker(QObject *parent = 0);

signals:
    void finished();

public slots:
    void calibrate();
};

#endif // CAMERACALIBRATIONWORKER_H
