#include "seconddialog.h"
#include "ui_seconddialog.h"
#include "ui_mainwindow.h"
#include <QDir>
#include <QImage>
#include <QImageReader>

secondDialog::secondDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::secondDialog)
{
    ui->setupUi(this);

}

void secondDialog::setLabelText(QStringList str){

    QList<QLabel*> LabelList;
    LabelList = findChildren<QLabel*>();
    int len = str.size();
    for(int i = 0;i<len;i++)
        LabelList.at(i)->setText(str.at(i));

}




void secondDialog::on_pushButton_clicked()
{
    QString path = "/home/robin/QtProjects/panoramas";
    QDir directory(path);
    QStringList images = directory.entryList(QStringList() << "*.png" << "*.PNG",QDir::Files);
    int len = images.size();
    printf("len : %d\n",len);
    QList<QImage> imageList ;
    QLabel label_list;


    for(int i=0;i<len;i++){
        QString final_file = path +"/"+images.at(i);
        QPixmap pix(final_file);

        labelImageList[i].setPixmap(pix);
        labelImageList[i].show();
    }


}

secondDialog::~secondDialog()
{
    delete ui;
}
