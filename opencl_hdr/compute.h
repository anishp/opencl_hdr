//
//  compute.h
//  opencl_hdr
//
//  Created by Anish Pednekar on 7/5/14.
//  Copyright (c) 2014 Anish Pednekar. All rights reserved.
//

int oclCompute(unsigned char* h_image, unsigned char* h_bChannel,
            unsigned char* h_gChannel, unsigned char* h_rChannel,
            unsigned int width, unsigned int height,
            float filter[], int filter_width);



int serialCompute(unsigned char* h_image, unsigned char* h_bChannel,
             unsigned char* h_gChannel, unsigned char* h_rChannel,
             unsigned int width, unsigned int height);
