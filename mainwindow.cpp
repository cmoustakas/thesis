#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "seconddialog.h"

#include <QMessageBox>
#include <QPixmap>
#include <QDir>
#include <QStringList>


#include <string.h>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>

#define TERMINATE '3'
#define MODE_2 '2'
#define MODE_1 '1'
#define UNDEFINED '0'

const int port_trans = 2288 ;
const int port_init = 5000;

char *droneIP = "192.168.2.7";
bool connection_established = false ;

char command[2] = {UNDEFINED,'\0'};
void transmit_commands(char code[2]);
bool capture_connection(struct timespec *wait_timeout);
void *connection_handler(void* t_args);
void try_connect(QWidget *parent,Ui::MainWindow *ui,QLabel *tick,QRadioButton *radio_Button_1,QRadioButton *radio_Button_2);
void update_dialog(secondDialog *dialog);

pthread_mutex_t calculating = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t done = PTHREAD_COND_INITIALIZER;


typedef struct arguments{
    int sock;
}args;



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    panoramas_dialog = new secondDialog(this);

    std::thread updater(update_dialog,panoramas_dialog);
    updater.detach();

    ui->radioButton_1->setEnabled(true);
    ui->radioButton_2->setEnabled(true);


    QMessageBox::about(parent,
                     "Connection:",
                     "Wait for capturing connection with drone");

    try_connect(this,ui,ui->tick,ui->radioButton_1,ui->radioButton_2);




    /** Init Logos **/
    QPixmap app("/home/robin/QtProjects/7.png");
    int w_app = ui->app_logo->width();
    int h_app = ui->app_logo->height();
    ui->app_logo->setPixmap(app.scaled(w_app,h_app,Qt::KeepAspectRatio));


    QPixmap auth_logo("/home/robin/QtProjects/auth.jpg");
    int w_a = ui->label_pic_1->width();
    int h_a = ui->label_pic_1->height();
    ui->label_pic_1->setPixmap(auth_logo.scaled(w_a,h_a,Qt::KeepAspectRatio));

    QPixmap poweroff("/home/robin/QtProjects/shutdown.png");
    int w_p = ui->shut_down->width();
    int h_p = ui->shut_down->height();
    ui->shut_down->setPixmap(poweroff.scaled(w_p,h_p,Qt::KeepAspectRatio));

    QPixmap cam("/home/robin/QtProjects/camera.png");
    int w_c = ui->stream->width();
    int h_c = ui->stream->height();
    ui->stream->setPixmap(cam.scaled(w_c,h_c,Qt::KeepAspectRatio));

    QPixmap panorama("/home/robin/QtProjects/pano.png");
    int w_pan = ui->pano->width();
    int h_pan = ui->pano->height();
    ui->pano->setPixmap(panorama.scaled(w_pan,h_pan,Qt::KeepAspectRatio));





}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_radioButton_1_clicked()
{
    panoramas_dialog->close();

    if(ui->radioButton_2->isChecked())
        ui->radioButton_2->setChecked(false);

    command[0] = MODE_1;
    command[1] = '\0';

    panoramas_dialog->close();

    std::thread transmit(transmit_commands,command);
    transmit.detach();

}

void MainWindow::on_radioButton_2_clicked()
{
    if(ui->radioButton_1->isChecked())
        ui->radioButton_1->setChecked(false);

    command[0] = MODE_2;
    command[1] = '\0';

    panoramas_dialog->show();

    std::thread transmit(transmit_commands,command);
    transmit.detach();


}





void MainWindow::on_pushButton_2_clicked()
{
    /** Description for Mode 1 **/
    QMessageBox::about(this,
                       "Description for Mode 1",
                        "By selecting Mode 1, jetson streams real time video captured by camera.In case of fire detection bounding boxes are generated.");

}

void MainWindow::on_pushButton_clicked()
{
    /** Description for Mode 2 **/
    QMessageBox::about(this,
                       "Description for Mode 2",
                       "By selecting Mode 2, jetson transmits panoramic frames to host in addition to GPS location when fire is spotted.");



}

void MainWindow::on_pushButton_3_clicked()
{
    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                             "Terminate",
                                                             "Are you sure you want to terminate process ?",
                                                              QMessageBox::Yes | QMessageBox::No);
    if(reply == QMessageBox::Yes){
        command[0] = TERMINATE;
        command[1] = '\0';
        transmit_commands(command);
        QApplication::quit();
    }

}







void transmit_commands(char code[2]){


    int sockfd = socket(AF_INET,SOCK_STREAM,0);

    struct sockaddr_in clientAddr;

    clientAddr.sin_family = AF_INET;
    clientAddr.sin_addr.s_addr = inet_addr(droneIP);
    clientAddr.sin_port = htons(port_trans);

    struct timeval timeout ;
    timeout.tv_sec = 10;
    timeout.tv_usec = 0;
    setsockopt(sockfd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(struct timeval));




    if(connect(sockfd,(struct sockaddr*)&clientAddr , sizeof(clientAddr)) == -1){
        printf("Couldn't connect \n");
        return ;
    }

    if(send(sockfd,code,2,0) == -1){
        printf("Couldn't send \n");
        return ;
    }

    close(sockfd);


}

bool capture_connection(struct timespec *wait_timeout){

    struct timespec abs_time;


    pthread_t con_t;
    int err;


    args *t_args = (args *)malloc(sizeof(struct arguments));

    int sockfd = socket(AF_INET,SOCK_STREAM,0);
    struct sockaddr_in serverAddress;

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddress.sin_port = htons(port_init);

    bind(sockfd,(struct sockaddr*)&serverAddress,sizeof(serverAddress));
    t_args->sock = sockfd;



    pthread_mutex_lock(&calculating);

    clock_gettime(CLOCK_REALTIME,&abs_time);
    abs_time.tv_sec += wait_timeout->tv_sec;
    abs_time.tv_nsec += wait_timeout->tv_nsec;

    pthread_create(&con_t,NULL,&connection_handler,(void*)t_args);

    err = pthread_cond_timedwait(&done,&calculating,&abs_time);

    if(err == ETIMEDOUT){
        close(sockfd);
        pthread_mutex_unlock(&calculating);
        return false;
    }
    else{
        close(sockfd);
        pthread_mutex_unlock(&calculating);
        return true;
    }


}

void *connection_handler(void* t_args){
     printf("In thread \n");

     int oldtype;
     pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,&oldtype);

    char receive[9];
    int sock = ((args*)t_args)->sock ;
    listen(sock,5);
    int clientSock = accept(sock,NULL,NULL);

    int ret = recv(clientSock,receive,9,0);

    printf("received :  %s \n",receive);
    if(ret != -1 && strcmp(receive,"connected")==0 )
        pthread_cond_signal(&done);


    return NULL;

}

void try_connect(QWidget *parent,Ui::MainWindow *ui,QLabel *tick,QRadioButton *radio_Button_1,QRadioButton *radio_Button_2){

    bool retry = true;

    struct timespec wait_timeout;
    memset(&wait_timeout,0,sizeof(wait_timeout));
    wait_timeout.tv_sec = 20;


    do{




        if(!capture_connection(&wait_timeout)){
            QMessageBox::StandardButton reply = QMessageBox::question(parent,
                                                                 "ERROR:",
                                                                 "Connection didn't established, make sure all connections are OK.\
                                                                  Do you want to Retry ?",
                                                                  QMessageBox::Yes | QMessageBox::No);


            if(reply==QMessageBox::Yes)
                retry = true;
            else{
                command[0] = TERMINATE;
                command[1] = '\0';
                QApplication::quit();
            }

        }
        else
           retry = false;


    }while(retry);


    QPixmap tick_logo("/home/robin/QtProjects/tick.png");
    int w_2 = tick->width();
    int h_2 = tick->height();
    tick->setPixmap(tick_logo.scaled(w_2,h_2,Qt::KeepAspectRatio));

    radio_Button_1->setEnabled(true);
    radio_Button_2->setEnabled(true);


}

void update_dialog(secondDialog *dialog){

    dialog->setWindowTitle("Panoramas Received ");
    int update_len=0;
    QDir directory("/home/robin/QtProjects/panoramas");

    while(command[0] != TERMINATE){
        QStringList images = directory.entryList(QStringList() << "*.png" << "*.PNG",QDir::Files);
        int len = images.size();

        while(command[0] == MODE_1 || command[0] == UNDEFINED){
            dialog->close();
            sleep(2);
        }

        if(command[0] == MODE_2 && update_len != len){
            update_len = len;

            dialog->setLabelText(images);


        }
        sleep(3);
    }

}

