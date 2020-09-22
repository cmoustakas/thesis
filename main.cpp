/**
    Remote Drone controller
**/
#include "mainwindow.h"
#include "PracticalSocket.h"

#include <QApplication>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>


#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <fcntl.h>

/** OpenCV host Libraries **/
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>


#define BACKLOG 1000

#define TERMINATE '3'
#define MODE_2 '2'
#define MODE_1 '1'
#define UNDEFINED '0'

#define PACK_SIZE 4096
#define BUF_LEN 65540


using namespace std;
using namespace std::chrono;
using namespace cv;

void parent_process(int fd[2]);
void child_process(int fd[2],int argc,char* argv[]);
void receive_data();
void receive_panorama(int sock);
void receive_location(int sock);
void transmit_commands(char command[2]);
void *handler(void *t_args);
void runGUI(int argc,char* argv[]);
void live_stream();


typedef struct arguments{
    int client_socket;
    int mode;

}args;

pthread_mutex_t mut;

bool joined=false;


int pano_received = 0;
int last_command=0;

const int port_pano_recv = 8000;
const int port_live_recv = 1935;
UDPSocket sock(port_live_recv);

extern char command[2];





int main(int argc, char *argv[])
{

    int status;
    pid_t p,wait_pid;
    int fd[2];

    if(pipe(fd) == -1){
        printf("Error in pipe \n");
        exit(0);
    }

    p = fork();

    if(p>0)
        child_process(fd,argc,argv);
    else if(p==0){
        parent_process(fd);
        while(1){
            wait_pid = waitpid(p,&status,WNOHANG);
            if(WIFEXITED(status) || WIFSIGNALED(status))
                break ;
            }
        if(WIFSIGNALED(status)){
            if(WCOREDUMP(status)){
                printf("core dumped  \n");
                exit(0);
            }
        }
        else
            printf(" status exited as : %d \n",WEXITSTATUS(status));

    }
    else
        printf("[-][-] Fork failed \n");

    return 0;
}

void parent_process(int fd[2]){
    /**
        Parent process is the
        constant reader of childs
        process updates
     **/

    std::thread receiver(receive_data);
    receiver.detach();
    std::thread live_thr(live_stream);
    live_thr.detach();


    char updated_command[2] = {UNDEFINED,'\0'};


    close(fd[1]);

    read(fd[0],updated_command,2);
    last_command = atoi(&updated_command[0]);
    printf("Updated command %s \n",updated_command);








    while(updated_command[0] != TERMINATE){
        read(fd[0],updated_command,2);
        last_command = atoi(&updated_command[0]);
        printf("Updated command %s \n",updated_command);

        sleep(1);


    }
}





void child_process(int fd[2],int argc,char*argv[]){
    /**
        Child process is the
        constant writer to the
        Parent process
    **/

    close(fd[0]); /** Close read end **/

    std::thread gui(runGUI,argc,argv);
    gui.detach();

    while(command[0] == UNDEFINED)
        sleep(1);


    write(fd[1],command,2);

    char temp_command[2];
    temp_command[0] = command[0];
    temp_command[1] = '\0';

    while(command[0] != TERMINATE){

        while(temp_command[0] == command[0]) /** While user keeps command unchanged stay in the loop **/
            usleep(200);

        write(fd[1],command,2);
        temp_command[0] = command[0];
    }

    exit(0);
}


void runGUI(int argc,char*argv[]){
    QApplication app(argc, argv);
    MainWindow w;

    w.show();

    int ret = app.exec();
    joined = true;
}


void receive_data(){


    int sockfd = socket(AF_INET,SOCK_STREAM,0);
    struct sockaddr_in serverAddress;

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddress.sin_port = htons(port_pano_recv);

    bind(sockfd,(struct sockaddr*)&serverAddress,sizeof(serverAddress));

    int n = std::thread::hardware_concurrency();

    /**
        Sometimes low-level programming
        is the only way gents
    **/

    pthread_t threads[100];




    listen(sockfd,BACKLOG);

    int thr_pointer = 0;

    printf("receiver \n");

    while(last_command == 0)
        usleep(500);


    while(last_command != 3){
        \
        while(last_command == 2){
            int clientSock = accept(sockfd,NULL,NULL);


            args *t_args = (args *)malloc(sizeof(struct arguments));


            t_args->client_socket = clientSock;
            t_args->mode = last_command ;





            if(thr_pointer>0)
                pthread_join(threads[thr_pointer-1],NULL);

            pthread_create(&threads[thr_pointer],NULL,&handler,(void*)t_args);
            thr_pointer++;



            if(thr_pointer == 100){
               for(int i=0;i<100;i++)
                   pthread_join(threads[i],NULL);

               thr_pointer=0;
            }

        }

        sleep(1);
    }


    close(sockfd);


}



void *handler(void *t_args){

    printf("\033[0m");
    printf ("\033[32;1m ");


    int clientSock = ((args*)t_args)->client_socket ;
    int op_mode = ((args*)t_args)->mode;




    char compressed_size[10];
    if(recv(clientSock,compressed_size,10,0) == -1){
        printf("Unable to receive compressed size \n");
        return NULL;
    }

    int size = atoi(compressed_size) ;

    unsigned char compressed_data_array[size];

    int bytes = 0;
    int total_bytes = 0;

    while(total_bytes < size){
        if((bytes = recv(clientSock,compressed_data_array+total_bytes,size-total_bytes,0)) == -1){
            printf("Unable to receive compressed frame \n");
            return NULL;
        }
        total_bytes = total_bytes+bytes;
    }

    std::vector<unsigned char>compressed_data ;

    for(int i=0;i<size;i++)
        compressed_data.push_back(compressed_data_array[i]);


    try{
        cv::Mat frame = imdecode(compressed_data,cv::IMREAD_COLOR);


        /**
            Set the title for storing panorama
        **/


        char title_png[50];
        strcpy(title_png,"/home/robin/QtProjects/panoramas/panorama");
        char num_of_panos[5];
        sprintf(num_of_panos,"%d",pano_received);
        strcat(title_png,num_of_panos);
        strcat(title_png,".png");
        imwrite(title_png,frame);

        printf("    [+][+] Panorama Saved : %s \n",title_png);

        pano_received++;

        frame.release();

    }
    catch(cv::Exception& e){
        if(last_command != 1 && last_command != 3){
            const char *err = e.what();
            printf("Error, Caught : %s \n",err);
        }
    }

    return NULL;
}

void live_stream(){
    namedWindow("live", WINDOW_AUTOSIZE);
        try {


            char buffer[BUF_LEN]; // Buffer for echo string
            int recvMsgSize; // Size of received message
            string sourceAddress; // Address of datagram source
            unsigned short sourcePort; // Port of datagram source


            while(last_command != 3){

                while (last_command == 1) {


                // Block until receive message from a client
                    do {
                        recvMsgSize = sock.recvFrom(buffer, BUF_LEN, sourceAddress, sourcePort);
                    } while (recvMsgSize > sizeof(int));
                    int total_pack = ((int * ) buffer)[0];

                    cout << "expecting length of packs:" << total_pack << endl;
                    char * longbuf = new char[PACK_SIZE * total_pack];
                    for (int i = 0; i < total_pack; i++) {
                        recvMsgSize = sock.recvFrom(buffer, BUF_LEN, sourceAddress, sourcePort);

                        if (recvMsgSize != PACK_SIZE)
                            continue;

                        memcpy( & longbuf[i * PACK_SIZE], buffer, PACK_SIZE);
                    }

                    cout << "Received packet from " << sourceAddress << ":" << sourcePort << endl;

                    Mat rawData = Mat(1, PACK_SIZE * total_pack, CV_8UC1, longbuf);
                    Mat frame = imdecode(rawData, IMREAD_COLOR);

                    if (frame.size().width == 0)
                        continue;

                    imshow("live", frame);
                    free(longbuf);

                    waitKey(1);

                }

                sleep(1);
            }
        }
        catch (SocketException & e) {
            cerr << e.what() << endl;
            exit(1);
        }



}
