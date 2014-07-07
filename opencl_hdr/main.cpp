//
//  main.cpp
//  opencl_hdr
//
//  Created by Anish Pednekar on 7/5/14.
//  Copyright (c) 2014 Anish Pednekar. All rights reserved.
//


#include <iostream>

#include "opencv2/opencv.hpp"
#include "compute.h"


using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    
    if (argc != 2) {
        cout<<"Wrong number of arguments!\n";
        return EXIT_FAILURE;
    }
    
    Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    
    unsigned int width = img.rows;
    unsigned int height = img.cols;
    
    // Single channel matrices to store separated
    Mat bChannel(width, height, CV_8UC1);
    Mat gChannel(width, height, CV_8UC1);
    Mat rChannel(width, height, CV_8UC1);
    
    // call the compute function which sets up and launches the openCL kernel
    if(oclCompute(img.data, bChannel.data, gChannel.data, rChannel.data, width, height)!=EXIT_SUCCESS)
        cout<<"Error!\n";
   
    /*
    if(serialCompute(img.data, bChannel.data, gChannel.data, rChannel.data, width, height)!=EXIT_SUCCESS)
        cout<<"Error!\n";
     */
    
    imwrite("b.jpg", bChannel);
    imwrite("g.jpg", gChannel);
    imwrite("r.jpg", rChannel);
    cout<<"\n";
    
    return 0;
}

