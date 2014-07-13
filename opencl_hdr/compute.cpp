//
//  compute.cpp
//  opencl_hdr
//
//  Created by Anish Pednekar on 7/5/14.
//  Copyright (c) 2014 Anish Pednekar. All rights reserved.
//


#include <stdio.h>
#include <OpenCL/OpenCL.h>
#include <unistd.h>

#define FILTER_SIZE 9


int oclCompute(unsigned char* h_image, unsigned char* h_bChannel,
            unsigned char* h_gChannel, unsigned char* h_rChannel,
            float filter[],
            unsigned int width, unsigned int height)
{
    
    int err;
    size_t global;  // global domain size
    size_t local;   // local domain size
    
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    
    cl_mem input;
    cl_mem d_bChannel;
    cl_mem d_gChannel;
    cl_mem d_rChannel;
    cl_mem d_filter;
    
    
    // load kernel source
    FILE *kern_f;
    size_t sourceSize;
    char* kernelSource;
    
    kern_f = fopen("./kernel.cl", "r");
    if(kern_f==NULL) {
        printf("Error: cannot open source file\n");
        return EXIT_FAILURE;
    }
    
    fseek(kern_f, 0, SEEK_END);
    sourceSize = ftell(kern_f);
    fseek(kern_f, 0, SEEK_SET);
    
    kernelSource = (char*) malloc(sourceSize + 1);
    fread(kernelSource, 1, sourceSize, kern_f);
    fclose(kern_f);
    
    
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err!=CL_SUCCESS) {
        printf("Error: failed to create a device group");
        return EXIT_FAILURE;
    }
    
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: failed to create device context\n");
        return EXIT_FAILURE;
    }
    
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: failed to create a command queue\n");
        return EXIT_FAILURE;
    }
    
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
    if(!program)
    {
        printf("Error: failed to create program from source\n");
        return EXIT_FAILURE;
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err!=CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        
        printf("Error: program build failed\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    
    kernel = clCreateKernel(program, "separateChannels", &err);
    if (!kernel || err!=CL_SUCCESS) {
        printf("Error: failed to create compute kernel\n");
        return EXIT_FAILURE;
    }
    
    
    input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * width * height * 3, NULL, NULL);
    d_bChannel = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, NULL);
    d_gChannel = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, NULL);
    d_rChannel = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height, NULL, NULL);
    d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * FILTER_SIZE, NULL, NULL);
    
    if(!input || !d_bChannel || !d_gChannel || !d_rChannel || !d_filter)
    {
        printf("Error: could not allocate device memory\n");
        return EXIT_FAILURE;
    }
    
    
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(unsigned char) * width * height * 3, h_image, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: could not copy data to device");
        return EXIT_FAILURE;
    }
    
    err = clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, sizeof(float) * FILTER_SIZE, filter, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Error: could not copy data to device");
        return EXIT_FAILURE;
    }
    
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_bChannel);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_gChannel);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_rChannel);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &width);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &height);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_filter);
    if (err!=CL_SUCCESS) {
        printf("Error: could not set kernel arguments\n");
        return EXIT_FAILURE;
    }
    
    
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: failed to retrieve workgroup infor\n");
        return EXIT_FAILURE;
    }
    else {
        printf("Workgroup size is %zu\n", local);
    }
    
    global = (width * height) + ((width * height)%local);
    
    size_t local2d[2] = {16, 16};
    size_t global2d[2] = {(width + width%local2d[0]), (height + height%local2d[1])};
    
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, (size_t *) global2d, (size_t *)local2d, 0, NULL, NULL);
    if (err) {
        printf("Error: failed to execute kernel\nError code: %d", err);
        return EXIT_FAILURE;
    }
    
    clFinish(commands);
    
    err = clEnqueueReadBuffer(commands, d_bChannel, CL_TRUE, 0, sizeof(unsigned char) * width * height, h_bChannel, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commands, d_gChannel, CL_TRUE, 0, sizeof(unsigned char) * width * height, h_gChannel, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(commands, d_rChannel, CL_TRUE, 0, sizeof(unsigned char) * width * height, h_rChannel, 0, NULL, NULL);
    if (err!=CL_SUCCESS) {
        printf("Error: failed to read output array\n");
        return EXIT_FAILURE;
    }
    
    
    clReleaseMemObject(input);
    clReleaseMemObject(d_bChannel);
    clReleaseMemObject(d_gChannel);
    clReleaseMemObject(d_rChannel);
    
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
}

int serialCompute(unsigned char* h_image, unsigned char* h_bChannel,
            unsigned char* h_gChannel, unsigned char* h_rChannel,
            unsigned int width, unsigned int height)
{
    for (int i=0; i<(width * height); i++) {
        h_bChannel[i] = h_image[i * 3];
        h_gChannel[i] = h_image[i * 3 + 1];
        h_rChannel[i] = h_image[i * 3 + 2];
    }
    return 0;
}

