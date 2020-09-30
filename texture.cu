/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (image_filename) and generates
 * an output PGM image (image_filename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "thanos.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

////////////////////////////////////////////////////////////////////////////////
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float* dData, int height, int width, float* mask, int masksize)
{
    // calculate normalized texture coordinates
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    int S = (masksize-1)/2;
    float sum = 0.0;
    int pixPos = row*width + col;
    dData[pixPos] = 0.0;

      for(int maskrow = -S; maskrow <= S; maskrow++){
        for(int maskcol = -S; maskcol <= S; maskcol++){
          int maskP = (maskrow+S)*masksize + (maskcol+S);
            sum += mask[maskP] * tex2D(tex, (col+maskcol+0.5f)/(float)width, (row+maskrow+0.5f)/(float)height);
        }
      }
      dData[pixPos] = sum;
      if (dData[pixPos] < 0){
        dData[pixPos] = 0;
      }
      else if(dData[pixPos] > 1){
        dData[pixPos] = 1;
      }
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    runTest(argc, argv);
    cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // int masksize = 3;
    // int masksize = 5;
    int masksize = 7;
    float* dFilter = NULL;
    int fsize =  masksize*masksize*sizeof(float);
    checkCudaErrors(cudaMalloc((void** )&dFilter,fsize));

    float hEdge3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float hSharpen3[] = {-1.0, -1.0, -1.0, -1.0, 9, -1.0, -1.0, -1.0, -1.0};
    float hAverage3[] = {0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111};
   //
    float hSharpen5[] = {-1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, 25, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0};
   // //
   float hAverage5[] = {0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,};

   float hSharpen7[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, 49, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0};
   //
   float hAverage7[] = {1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                        1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49};

    checkCudaErrors(cudaMemcpy(dFilter, hAverage7, fsize, cudaMemcpyHostToDevice));

    // Allocate device memory for result
    float *dData = NULL;

    checkCudaErrors(cudaMalloc((void **) &dData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      hData,
                                      size,
                                      cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Warmup
    // transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

    checkCudaErrors(cudaDeviceSynchronize());

    // Execute the kernel
    StopWatchInterface *timer2 = NULL;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);
    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, dFilter, masksize);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer2);
    float t2 = sdkGetTimerValue(&timer)/1000.0f;
    printf("Texture memory took %f  \n", t2);
    sdkDeleteTimer(&timer2);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_TEXedge.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    sdkStopTimer(&timer);
    float t = sdkGetTimerValue(&timer) / 1000.0f;
    printf("Texture overhead took %f time \n", t-t2);
    sdkDeleteTimer(&timer);

    // Write regression file if necessary
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // Write file for regression test
        sdkWriteFile<float>("./data/regression.dat",
                            hOutputData,
                            width*height,
                            0.0f,
                            false);
    }
    else
    {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
    }

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
    // free(refPath);
}
