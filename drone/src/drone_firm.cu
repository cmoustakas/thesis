/**

MIT License

Copyright (c) 2020 Chares Moustakas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**/


/** 
	


	[+] Author : Chares Moustakas.
	[+] E-mail : cmoustakas@ece.auth.gr , charesmoustakas@gmail.com 
	[+] Professor : Nikolaos Pitsianis,Dimitrios Floros. 
	[+] University : Aristotle's University Of Thessaloniki 
	[+] Department : Electrical and Computer Engineering 


	[+] Thesis Description : 
		Based On detect-net Nvidia's Algorithm the code below loads a pretrained (dataset  based on fires) 
		neural network, send location of the fire that was detected and  additionally it provides stitched images for further 
		in-depth image recognition on a parallel computing concept.



**/


/** 
	C++ libraries 
**/
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>

/** 
	Thanks to :  
		http://cs.ecs.baylor.edu/~donahoo/practical/CSockets/practical/
**/
#include "PracticalSocket.h"


/**
	Traditional C libraries 
**/
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <malloc.h>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>


/**
	CUDA liraries
**/
#include <cuda.h>
#include <cuda_runtime.h>

/** 
	Jetson Inference & Utils libraries
**/
#include <jetson-utils/gstCamera.h>
#include <jetson-inference/detectNet.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/cudaRGB.h>


/** 
	OpenCV host libraries 
**/
#include <opencv2/core/version.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>


/** 
	OpenCV CUDA libraries 
**/
#include <opencv2/core/cuda.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>




/** Namespaces **/
using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace cv::cuda;
using namespace cv::xfeatures2d;


/** Constants **/
#define RIGHT 0
#define LEFT  2
#define UP -1
#define DOWN 1

#define BACKLOG 5

#define UNDEFINED 0

#define MODE_1 1 	//Sending only location 
#define MODE_2 2 	// Sending location and panoramic frames 
#define TERMINATE 3 // Terminate process

#define FRAME_INTERVAL (1000/30)
#define PACK_SIZE 4096 //udp pack size; note that OSX limits < 8100 bytes
#define ENCODE_QUALITY 80

/** Function's Decl. **/

/** 
	Thread Handlers 
**/

// C - style posix threads 
void *panorama_handler(void *data);
void *optical_flow_handler(void *data);
void *live_handler(void *data);

// C++ - style posix threads
void receive_commands();
void transmit_output(Mat &frame,int mode);
void gpu_thread_tracker();
bool init_connection_with_host();

// Functions :
void send_frame(int sockfd,Mat &frame);
void decideDirection(const vector<Point2f>& prevPts,const vector<Point2f>& nextPts,const vector<uchar>&status);
void downloadP(const GpuMat& d_mat, vector<Point2f>& vec);
void downloadU(const GpuMat& d_mat, vector<uchar>& vec);
void calculate_mem_threshold();
int nearestTo(double direction,int theta1,int theta2);
void reset_bools();


/** Global Variables **/
unsigned int transmitions = 0;
int dir,displ;
int controll_mode = UNDEFINED;

const int throuput_memory_threshold = 95; // 95% memory usage threshold
const int FPS_PENALTY=70; // Decide direction total time = 2.5 seconds , FPS = 25 => frame penalty =~ 60

const int port_recv = 2288 ;
const int port_trans_pano = 8000;
const int port_init = 5000;

unsigned short servPort = Socket::resolveService("1935", "udp");
UDPSocket sock;

const int height = 338;
const int width = 640;

bool signal_recieved = false;
bool predict_direction_flag = false;
bool spy_frames = false;
bool thread_alive_flag = false;
bool transmit_pano_flag = false;
bool rotation = false;
bool power_save_mode = false;

const char *hostIP = "192.168.2.10";



vector < uchar > encoded;
vector < int > compression_params;


GpuMat d_frame1;

/** 
	Struct for panorama thread 
**/
typedef struct arguments_p{
	float *t_leftRGBA;
	float *t_rightRGBA;
	int t_rows;
	int t_cols;
	bool t_penalty;
}pano_args;

pthread_t pano_thread;

/**
	Struct for optical flow thread
**/

typedef struct arguments_of{
	float *t_leftRGBA;
	float *t_rightRGBA;
	int t_rows;
	int t_cols;
}optical_flow_args;

pthread_t optical_flow_thread;

/**	
	Struct for transmition 
	
**/ 

typedef struct arguments_live{
	uchar3 *frame;
	int width;
	int height;
}live_stream_args;


pthread_t transmitter_thread;


void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: detectnet-camera [-h] [--network NETWORK] [--threshold THRESHOLD]\n");
	printf("                        [--camera CAMERA] [--width WIDTH] [--height HEIGHT]\n\n");
	printf("Locate objects in a live camera stream using an object detection DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network NETWORK pre-trained model to load (see below for options)\n");
	printf("  --overlay OVERLAY detection overlay flags (e.g. --overlay=box,labels,conf)\n");
	printf("                    valid combinations are:  'box', 'labels', 'conf', 'none'\n");
     printf("  --alpha ALPHA     overlay alpha blending value, range 0-255 (default: 120)\n");
	printf("  --camera CAMERA   index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("                    or for VL42 cameras the /dev/video device to use.\n");
     printf("                    by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH     desired width of camera stream (default is 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of camera stream (default is 720 pixels)\n");
	printf("  --threshold VALUE minimum threshold for detection (default is 0.5)\n\n");

	printf("%s\n", detectNet::Usage());

	return 0;
}





int main( int argc, char** argv )
{
	

	static int PREALLOC_SIZE     = 300 * 1024 * 1024; // Preallocate 300MB for our Process [+]

	// Disable paging for the current process 
	mlockall(MCL_CURRENT | MCL_FUTURE);				// forgetting munlockall() when done!

	// Turn off malloc trimming AKA, leave the heap alone . 
	mallopt(M_TRIM_THRESHOLD, -1);

	// Virtual RAM disabling . 
	mallopt(M_MMAP_MAX, 0);

	unsigned int page_size = sysconf(_SC_PAGESIZE);
	unsigned char * buffer = (unsigned char *)malloc(PREALLOC_SIZE);

	/** Touch each page in this piece of memory to get it mapped into RAM **/
	for(int i = 0; i < PREALLOC_SIZE; i += page_size)
		buffer[i] = 0; /** This will generate pagefaults that will provide larger portions of RAM to our Process by given one and only large Page **/
	
	
	free(buffer);

	

	while(!init_connection_with_host())
		sleep(1);

	std::thread remote_controller(receive_commands);
	remote_controller.detach();

	while(controll_mode == UNDEFINED)
		sleep(1);


	std::thread tracker_thr(gpu_thread_tracker);
	tracker_thr.detach();


	/** Parse CMD **/
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();


	/** Attach signal handler **/
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	
 
	
	gstCamera* camera = gstCamera::Create(width,height,cmdLine.GetString("camera"));

	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	
	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to load detectNet model\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlaySpyFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,conf"));
	/** Different flags because bounding boxes add noise to stitching algorithm **/
	const uint32_t flagsForStitch = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "none"));


	if( !camera->Open() )
	{
		printf("detectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("detectnet-camera:  camera open for streaming\n");
	
	
	
	float *leftImgRGBA = NULL;
	float *cyrcleFrameBuffer[6];

	int  frameBound;
	int frameCount=0,numDetections=0,bandwidth = 0,frameBuffCnt=0,counter=0;
	
	int* directionPtr = (int*)malloc(2*sizeof(int));
	bool penalty = false;
	
	
	

	live_stream_args *l_args = (live_stream_args*)malloc(sizeof(struct arguments_live));
	pano_args *p_args = (pano_args *)malloc(sizeof(struct arguments_p));
	optical_flow_args *of_args = (optical_flow_args*)malloc(sizeof(struct arguments_of));
	
	unsigned int sync_flag = cudaDeviceScheduleBlockingSync ;

	cudaSetDeviceFlags(sync_flag);


	while( !signal_recieved )
	{
		
		
		/** capture RGBA image **/
		float *imgRGBA = NULL;
		
			

		if( !camera->CaptureRGBA(&imgRGBA,1000,false) ){
			printf("detectnet-camera:  failed to capture RGBA frame from camera \n");
			while(thread_alive_flag) /** Wait for my threads to join **/
				usleep(400);
			
			break ;
		}

		frameBuffCnt++;
		
		/** Save 1 Image per 10 frames (for 60 frames = FPS_PENALTY) overlap **/ 
		if(frameBuffCnt == 10){
			cyrcleFrameBuffer[counter] = imgRGBA;
			counter++;
			frameBuffCnt = 0;

			if(counter == 6)
				counter = 0;
		}

		if(controll_mode == MODE_2 && !transmit_pano_flag){
			
			

			/** If Network Recognized Somethin Start Stitching Process **/ 
			if( !spy_frames && !predict_direction_flag && !thread_alive_flag && numDetections > 0 ){
				/** capture two respective frames **/
				
				camera->CaptureRGBA(&imgRGBA,1000,false);
				of_args->t_leftRGBA = imgRGBA;
				
				float *imgRGBA = NULL;
				
				camera->CaptureRGBA(&imgRGBA,1000,false);
				of_args->t_rightRGBA = imgRGBA;			
				
				of_args->t_rows = height;
				of_args->t_cols = width;

				predict_direction_flag = true;
				
			}
			else if(predict_direction_flag){
				
				pthread_create(&optical_flow_thread,NULL,&optical_flow_handler,(void*)of_args);
		
				spy_frames = true;
				predict_direction_flag = false;

			}
			else if(spy_frames && !thread_alive_flag){
				if(frameCount == 0){ /** Init frameBound and rotate, if its first time in the Club  **/
				
					if(dir == UP || dir == DOWN){
						frameBound =  (height)/(displ)  - 2;
						rotation = true;
					
					
					}
					else 
						frameBound =  (width)/(displ)  - 2;
					
				

					frameBound = frameBound/2;  /** I need A Good Enough Overlap Area **/

					if(frameBound > FPS_PENALTY){
					
						bandwidth = (frameBound - FPS_PENALTY)/4 ;
						if(bandwidth < 100)
							bandwidth=100;
						penalty = false;
					
					}
					else if(frameBound < FPS_PENALTY){
					
						penalty = true;
						bandwidth = (int)((frameBound)/10); 
						leftImgRGBA  = cyrcleFrameBuffer[counter-1];
						

					}
				
				}
			
				/** Bandwidth == Overlap Area Between Frames **/
				if(frameCount == bandwidth ){

					
					p_args->t_penalty = penalty;
					p_args->t_rows = height;
					p_args->t_cols = width;

					
					p_args->t_leftRGBA = leftImgRGBA;
					p_args->t_rightRGBA = imgRGBA;
					
					
					pthread_create(&pano_thread,NULL,&panorama_handler,(void*)p_args);


					leftImgRGBA = NULL;
					
					spy_frames = false ;
					frameCount = -1;
				}
				frameCount++;
				
			}
		}
		else if(controll_mode == TERMINATE){
			while(thread_alive_flag) /** Wait all running threads to join **/
				usleep(200);
			printf("[-][-] Terminating Process [-][-] \n");
			break;
		}	
		

		detectNet::Detection* detections = NULL;
		
		/** Look after Launch failure **/
		
		if(!thread_alive_flag || power_save_mode ){
			if(controll_mode == 2)
				numDetections = net->Detect(imgRGBA, width, height, &detections, flagsForStitch);
			else if(controll_mode == 1)
				numDetections = net->Detect(imgRGBA, width, height, &detections, overlaySpyFlags);
		}

		

		if(controll_mode == MODE_1 && !thread_alive_flag ){
			
			uchar3 *stream_BGR_d = NULL;


			cudaMalloc((void**)&stream_BGR_d,width*height*sizeof(uchar3));
			
			cudaRGBA32ToBGR8( (float4*)imgRGBA, stream_BGR_d,(size_t)width,(size_t)height);
			
			uchar3 *stream_BGR_h;
			stream_BGR_h = (uchar3*)malloc(width*height*sizeof(uchar3));
			cudaMemcpy(stream_BGR_h,stream_BGR_d,width*height*sizeof(uchar3),cudaMemcpyDeviceToHost);
			
			l_args->width = width;
			l_args->height = height;
			l_args->frame = stream_BGR_h;
			
			/**
			pthread_join(transmitter_thread,NULL);
			pthread_create(&transmitter_thread,NULL,&live_handler,(void*)l_args);
			**/

			live_handler((void*)l_args);
			cudaFree((void*)stream_BGR_d);
			
			
		}
		

			
	}
		
	SAFE_DELETE(camera);
	SAFE_DELETE(net);


	return 0;
}





void decideDirection(const vector<Point2f>& prevPts,const vector<Point2f>& nextPts,const vector<uchar>&status){
    double direction,host=0,theta,hypotenuse ;
    int* retPtr = (int*)malloc(2*sizeof(int));
    for(size_t i =0;i<prevPts.size();i++){
        if(status[i]){
            host++;
            Point p = prevPts[i];
            Point n = nextPts[i];
            
            // Because of 1:1 tangent's nature there is no possible duplicate scenario 
            theta = atan2((double)p.y - n.y,(double)p.x-n.x);
            theta = theta*180/CV_PI ;  
            direction += theta;
            hypotenuse += sqrt( (double)(p.y - n.y)*(p.y - n.y) + (double)(p.x - n.x)*(p.x - n.x) );

            
		}

    }
    
    direction = direction/host;

    //Average Displacement Calculation 
    double averageDispl = hypotenuse/host;
    printf("Average Displacement : %d \n",(int)averageDispl);

    int quant;


    /** Quantize Angles as {0,90,180,-90} [+][+][+][+][+] **/

    if(direction < 0){
        quant = nearestTo(direction,-90,-180);
        if(quant == 1)
            quant = -quant;    
    }
    else if(direction >0)
        quant = nearestTo(direction,90,180);
    
   
    
    printf("[+][+] Direction decided, camera is moving :  ");
    
	if(quant == 1)
		printf("DOWN \n");
	else if(quant == -1)
		printf("UP \n");
	else if(quant == 0)
		printf("RIGHT \n");
	else 
		printf("LEFT \n");
    
    dir = quant;
	
	if(averageDispl < 1)
		displ = 1; /** Give Some Displacement For Pretty Slow Velocity **/
	else
		displ = averageDispl;
    
    
}

int nearestTo(double direction,int theta1,int theta2){
    
    int distance1 = abs(direction);
    int distance2 = abs(abs(direction) - abs(theta1));
    int distance3 = abs(abs(direction) - abs(theta2));
    int min = distance1;
    int ret = 0;

    if(distance2<min){
        min = distance2;
        ret = 1;
    }
    if(distance3 < min){
        min = distance3;
        ret = 2;
    }
    return ret ;

}

void downloadP(const GpuMat& d_mat, vector<Point2f>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

void downloadU(const GpuMat& d_mat, vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

void send_frame(int sockfd,Mat &frame){
	
	
	try{

		std::vector<unsigned char>compressed_data;
		imencode(".png",frame,compressed_data);

		char total_zip_bytes[10];
		sprintf(total_zip_bytes,"%d",compressed_data.size() );

		printf("Compressed Panorama BYTES : %s \n",total_zip_bytes);
	

		if(send(sockfd,total_zip_bytes,10,0) == -1){
			printf("Unable to send size \n");
			return ;
		}	
	
	
		int bytes_sent = 0 ;

	
		
		if(bytes_sent = send(sockfd,compressed_data.data(),compressed_data.size() ,0) == -1){
			printf("Unable to sent compressed data \n ");
			return ;
		}
	
		printf("[+][+][+] Sent compressed panorama [+][+][+] \n");
	}
	catch(cv::Exception &e){
		const char *err = e.what();
		printf("[send_frame] Encoding Error Caught [-][-] : \n");
		printf("	%s \n",err);
	}

	transmit_pano_flag = false;
 
   
    
}


bool init_connection_with_host(){
	
	char *init_message = "connected";

	int sockfd=socket(AF_INET,SOCK_STREAM,0);

	struct sockaddr_in clientAddr;
	
	clientAddr.sin_family = AF_INET;
	clientAddr.sin_addr.s_addr = inet_addr(hostIP);
	clientAddr.sin_port = htons(port_init);

	struct timeval timeout;
	timeout.tv_sec = 10;
	timeout.tv_usec = 0;
	setsockopt(sockfd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(struct timeval));

	if(connect(sockfd,(struct sockaddr*)&clientAddr,sizeof(clientAddr)) == -1 ){
		printf("Unable to connect \n");
		return false;
		
	}

	if(send(sockfd,init_message,9,0) == -1){
		printf("Unable to send size \n");
		return false;
	}
	printf("Init Connection with host : %s \n",hostIP);
			
	close(sockfd);
	return true;
}





void calculate_mem_threshold(){
	size_t free_mem;
	size_t total_mem;
	
	cudaMemGetInfo(&free_mem,&total_mem);	/** Get total and free memory amount. **/
	
	int usage_memory_per = (int)(100*float(total_mem-free_mem)/float(total_mem));

	if(usage_memory_per > throuput_memory_threshold){
		controll_mode = MODE_1;
		power_save_mode = true;
	}
	else
		power_save_mode = false;
		
	
		
}



void reset_bools(){
	
	predict_direction_flag = false;
	spy_frames = false;
	thread_alive_flag = false;
	rotation = false;

}

/** 
	My thread Functions [+][+]
	
		1 stitching function 
		2 decide direction via optical flow function
		3 server-listenner for commands 
		4 client-sender based on modes 

**/


void *panorama_handler(void *data){
	
	thread_alive_flag = true;
	
	float *rightRGBA = ((pano_args*)data)-> t_rightRGBA;
	float *leftRGBA = ((pano_args*)data)-> t_leftRGBA;
	
	int rows = ((pano_args*)data)->t_rows;
	int cols = ((pano_args*)data)->t_cols;
	
	bool penalty = ((pano_args*)data)->t_penalty;


	
	try{

		printf("[+][+][+][+][+][+][+][+][+]  Stitching Process Just Started [+][+][+][+][+][+][+][+][+] \n");

		uchar3 *rightBGR;
		cudaMalloc((void**)&rightBGR,rows*cols*sizeof(uchar3));
		cudaRGBA32ToBGR8( (float4*)rightRGBA, rightBGR,(size_t)cols,(size_t)rows);

		uchar3 *leftBGR;
		if(penalty){
		
			cudaMalloc((void**)&leftBGR,rows*cols*sizeof(uchar3));
			cudaRGBA32ToBGR8( (float4*)leftRGBA, leftBGR,(size_t)cols,(size_t)rows);
			GpuMat temp(rows,cols,CV_8UC3,(void*)leftBGR);
			temp.copyTo(d_frame1);
		}

		GpuMat gpu_mask1,gpu_mask2;
	
		GpuMat d_frame2(rows,cols,CV_8UC3,(void*)rightBGR);
	

		cv::cuda::Stream rot[2];
		if(rotation){
			int degr;
			if(dir == UP)
				degr = -90;
			else 
				degr = 90;

			cv::cuda::rotate( d_frame1, d_frame1, cv::Size(rows,cols ), degr, rows, 0, cv::INTER_LINEAR);
			cv::cuda::rotate( d_frame2, d_frame2, cv::Size(rows,cols ), degr, rows, 0, cv::INTER_LINEAR);
		}
	
		cuda::cvtColor(d_frame1,gpu_mask1,COLOR_BGR2GRAY,1);
    	cuda::cvtColor(d_frame2,gpu_mask2,COLOR_BGR2GRAY,1);

		Mat frame1(d_frame1);
		Mat frame2(d_frame2);
	
		
    	//--Step 1 : Detect the keypoints using SURF Detector
    	SURF_CUDA surf;

	    GpuMat keypoints1GPU, keypoints2GPU;
    	GpuMat descriptors1GPU, descriptors2GPU;

    	surf(gpu_mask1,GpuMat(),keypoints1GPU,descriptors1GPU);
    	surf(gpu_mask2,GpuMat(),keypoints2GPU,descriptors2GPU);


		/** Match Descriptors **/
    	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    	vector<DMatch> matches;
    	matcher->match(descriptors1GPU, descriptors2GPU,matches);
		

    	double max_dist = 0;
    	double min_dist = 100;


    	vector<KeyPoint> keypoints1, keypoints2;
    	vector<float> descriptors1, descriptors2;
    	surf.downloadKeypoints(keypoints1GPU, keypoints1);
    	surf.downloadKeypoints(keypoints2GPU, keypoints2);
    
    	/** Download gpu keypoints and descriptors **/
    	/** Because of the small size of each operation, gpu use is not worth anymore**/
    	/** ---------   **/

    	Mat descr1;
    	descriptors1GPU.download(descr1);

    	/** Here Stops GPU and Starts CPU job **/ 

    
    	int r = descr1.rows;

    	for(int i =0; i < descr1.rows ; i++)
    	{
        	double dist = matches[i].distance;
        	if(dist == 0.f){
            	r=i;
            	break;
        	}

        	if( dist < min_dist && dist > 0 ) min_dist = dist;
        	else if( dist > max_dist ) max_dist = dist;
    	}


		std::vector< DMatch > good_matches;


		/** Keep Only linear matches,because of vertical footage ,non-linear matches act like noise in the system **/

    	for(int i =0 ; i < r ; i++)
    	{
        
        	int idx2 = matches[i].trainIdx;
        	int idx1 = matches[i].queryIdx;
       
        	int theta = atan2((double)(keypoints2[idx2].pt.y- keypoints1[idx1].pt.y),(double)(keypoints2[idx2].pt.x-keypoints1[idx1].pt.x + frame1.cols))*180/CV_PI ;
    
        	if(  abs(theta)<10 && matches[i].distance < 4*min_dist) {
            	good_matches.push_back( matches[i] );
        	}
        
    	}



    	vector< Point2f > obj;
    	vector< Point2f > scene;

    	for( int i = 0; i < good_matches.size(); i++)
    	{
    	    //--Get the keypoints from the good matches
        	obj.push_back( keypoints1[good_matches[i].queryIdx].pt );
        	scene.push_back( keypoints2[good_matches[i].trainIdx].pt );
    	}

    	// Homography Matrix
    	Mat H = findHomography(scene,obj,RANSAC);
    
    	/** Use the homography Matrix to warp the images **/
    	Mat result;
    	cv::warpPerspective(frame2, result, H, cv::Size(frame1.cols+frame2.cols,frame1.rows));
   
    	Mat half(result, cv::Rect(0, 0, frame2.cols, frame2.rows) );
  
    	frame1.copyTo(half);

   		transmit_pano_flag = true;

		transmit_output(result,controll_mode);
	
		cudaFree((void*)rightBGR);
		cudaFree((void*)leftBGR);


	}
	catch(cv::Exception& e){
		const char* err_msg = e.what();
		printf("Caught error in stitching process ... retry :  %s ",err_msg);
		reset_bools();

	}
	
	thread_alive_flag = false;

	

}


void *optical_flow_handler(void *data){

	thread_alive_flag = true ;


	float *img2 = ((optical_flow_args*)data)-> t_rightRGBA;
	float *img1 = ((optical_flow_args*)data)-> t_leftRGBA;
	
	int rows = ((optical_flow_args*)data)->t_rows;
	int cols = ((optical_flow_args*)data)->t_cols;
	


	try{

		printf("[+][+][+][+][+][+][+][+][+]  Optical Flow Process Just Started [+][+][+][+][+][+][+][+][+] \n");



		auto t_start = std::chrono::high_resolution_clock::now();

		cv::cuda::Stream stream1,stream2,stream3;

		/** Transform RGBA 32F image to BGR 8U **/
		uchar3* output1 = NULL;
		uchar3 *output2 = NULL;
		cudaMalloc((void**)&output1,rows*cols*sizeof(uchar3));
		cudaMalloc((void**)&output2,rows*cols*sizeof(uchar3));

	
		cudaRGBA32ToBGR8( (float4*)img1, output1,(size_t)cols,(size_t)rows);
		cudaRGBA32ToBGR8( (float4*)img2, output2,(size_t)cols,(size_t)rows);
	


		GpuMat d_gray;
		GpuMat dev_frame1(rows,cols,CV_8UC3,(void*)output1);
		GpuMat dev_frame2(rows,cols,CV_8UC3,(void*)output2);
		GpuMat dev_prevPts;
	

		Mat frame1(dev_frame1);
		Mat frame2(dev_frame2);
	


		/** Create a copy for left frame of image stitching : d_frame1 **/
		dev_frame2.copyTo(d_frame1);


		cv::cuda::cvtColor(dev_frame1,d_gray,COLOR_BGR2GRAY,1,stream3);
	
		/** Detect Good Corners to Track **/
		//cudaDeviceSynchronize();
		Ptr<cuda::CornersDetector> detector = cuda::createGoodFeaturesToTrackDetector(d_gray.type(), 500, 0.01, 1);
    	detector->detect(d_gray, dev_prevPts,cv::noArray(),stream1);
	
	

    	GpuMat dev_nextPts;
    	GpuMat dev_status;
	
	

		/** Execute sparse optical Flow calculation **/
		//cudaDeviceSynchronize();
    	Ptr<cuda::SparsePyrLKOpticalFlow> opticalFlowObj = cuda::SparsePyrLKOpticalFlow::create(Size(21,21),3,30,false);
    	opticalFlowObj->calc(dev_frame1,dev_frame2,dev_prevPts,dev_nextPts,dev_status,cv::noArray(),stream2);

	
		/** Download previous, next Points and status for success or not **/
    	vector<Point2f> prevPts(dev_prevPts.cols);
    	downloadP(dev_prevPts, prevPts);

    	vector<Point2f> nextPts(dev_nextPts.cols);
    	downloadP(dev_nextPts, nextPts);

    	vector<uchar> status(dev_status.cols);
    	downloadU(dev_status, status);


		decideDirection(prevPts,nextPts,status);


		cudaFree((void*)output1);
		cudaFree((void*)output2);


	}
	catch(cv::Exception& e){
		const char *err_msg = e.what();
		printf("Error Caught in Optical Flow process : %s \n",err_msg);
		reset_bools();

	}

	thread_alive_flag = false;
	
	

}

/** 
	Receive commands thread-method , communicates with host via port 2288 
	Sends back data based on operation mode via port 8080
**/


void receive_commands(){
	
	char code[2];

	int sockfd = socket(AF_INET,SOCK_STREAM,0);
	struct sockaddr_in serverAddress;

	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = INADDR_ANY;
	serverAddress.sin_port = htons(port_recv);

	bind(sockfd,(struct sockaddr*)&serverAddress,sizeof(serverAddress));
	while(1){
		listen(sockfd,BACKLOG);

		int clientSock = accept(sockfd,NULL,NULL);
		recv(clientSock,code,2,0);

		printf("\n\n\n CONTROL MODE SET :  %s \n\n\n",code);

		controll_mode = atoi(&code[0]);
		
		if(controll_mode == 3)
			break;
		

	}

	close(sockfd);
	transmit_pano_flag = false;

}



/** 
	Thread that transmits the output based on the mode 
	that user selected from the remote machine 
**/
void *live_handler(void *data){
	
	uchar3 *stream_BGR = ((live_stream_args*)data)->frame;
	int width = ((live_stream_args*)data)->width;
	int height = ((live_stream_args*)data)->height;

	Mat frame(height,width,CV_8UC3,(void*)stream_BGR);

	
	int jpegqual =  ENCODE_QUALITY;

	
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(jpegqual);

    imencode(".jpg", frame, encoded, compression_params);

    int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;

    int ibuf[1];
    ibuf[0] = total_pack;
    sock.sendTo(ibuf, sizeof(int), hostIP, servPort);

    for (int i = 0; i < total_pack; i++)
        sock.sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, hostIP, servPort);

	waitKey(FRAME_INTERVAL);

	frame.release();
	free(stream_BGR);

}



void transmit_output(Mat &frame,int mode){
	
	
	int sockfd=socket(AF_INET,SOCK_STREAM,0);

	struct sockaddr_in clientAddr;
	
	clientAddr.sin_family = AF_INET;
	clientAddr.sin_addr.s_addr = inet_addr(hostIP);
	clientAddr.sin_port = htons(port_trans_pano);

	struct timeval timeout;
	timeout.tv_sec = 5;
	timeout.tv_usec = 0;
	setsockopt(sockfd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(struct timeval));

	if(connect(sockfd,(struct sockaddr*)&clientAddr,sizeof(clientAddr)) == -1 ){
		printf("Couldnt connect \n");
		int error ;
		socklen_t len = sizeof (error);
        int retval = getsockopt (sockfd, SOL_SOCKET, SO_ERROR, &error, &len);
		if(error != 0)
			fprintf(stderr, "\n\n socket error: %s \n\n", strerror(error));
		sleep(5);
		close(sockfd);
		
		return ;
	}
	else if(mode == controll_mode){
		
		send_frame(sockfd,frame);

	}
	frame.release();
	
	close(sockfd);
		


}


/** 

	Thread that tracks memory's throughput 
	if for some reason memory's usage grows at higher 
	level than 95% of total memory Mode 1 imidiately sets 
	until the above condition got false 

**/

void gpu_thread_tracker(){
	while(controll_mode != TERMINATE ){
		calculate_mem_threshold();
		usleep(200);
	}
}
