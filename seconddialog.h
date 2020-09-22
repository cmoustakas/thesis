#ifndef SECONDDIALOG_H
#define SECONDDIALOG_H

#include <QDialog>
#include <QMainWindow>
#include "ui_mainwindow.h"

namespace Ui {
class secondDialog;
}

class secondDialog : public QDialog
{
    Q_OBJECT

public:
    explicit secondDialog(QWidget *parent = nullptr);
    void setLabelText(QStringList str);
    QLabel labelImageList[10];


    ~secondDialog();

private slots:

    void on_pushButton_clicked();

private:
    Ui::secondDialog *ui;

};


#endif // SECONDDIALOG_H
