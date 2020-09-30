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
#include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f
// #define filtsize 3
 // #define filtsize 5
#define filtsize 7

// __device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
// __device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0};
__device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


const char *imageFilename = "thanos.pgm";

//Function headers
__global__ void convoluteConst(float* dData, float* hData, int height, int width);

int main(int argc, char **argv)
{
  //Load image
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
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

    float hEdge3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float hSharpen3[] = {-1.0, -1.0, -1.0, -1.0, 9, -1.0, -1.0, -1.0, -1.0};
    float hAverage3[] = {0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111};

    float hSharpen5[] = {-1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, 25, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0};

    float hAverage5[] = {0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,
                        0.04, 0.04, 0.04, 0.04, 0.04,};
    //
    float hSharpen7[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, 49, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0};


    float hAverage7[] = {1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49};

    float* dData = NULL;
    float* dOut = NULL;
    checkCudaErrors(cudaMalloc((void** )&dData, size));
    checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void** )&dOut, size));

    int fsize =  filtsize*filtsize*sizeof(float);
    checkCudaErrors(cudaMemcpyToSymbol(filter, hAverage7, fsize));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(height/dimBlock.x, width/dimBlock.y, 1);

    checkCudaErrors(cudaDeviceSynchronize());

    StopWatchInterface *timer2 = NULL;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);
    convoluteConst<<<dimGrid, dimBlock>>>(dOut, dData, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer2);
    float t2 = sdkGetTimerValue(&timer2)/1000.0f;
    printf("Constant took %f time \n",t2);
    sdkDeleteTimer(&timer2);

    float* hOut = (float* )malloc(size);
    cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);

    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 6, "_ConstAverage7.pgm");
    sdkSavePGM(outputFilename, hOut, width, height);
    printf("Wrote '%s' to file\n", outputFilename);

    sdkStopTimer(&timer);
    float t = sdkGetTimerValue(&timer) / 1000.0f;
    printf("Constant overhead took %f time \n", t-t2);
    sdkDeleteTimer(&timer);

}

///////////////////////////////////////
///////////// CONVOLUTIONS ///////////
/////////////////////////////////////
__global__ void convoluteConst(float* dData, float* hData, int height, int width){
  unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

  int S = (filtsize-1)/2;
  float sum = 0.0;
  int pixPos = row*width + col;
  // dData[pixPos] = 0.0;
  if(row<height && col<width){
    for(int maskrow = -S; maskrow <= S; maskrow++){
      for(int maskcol = -S; maskcol <= S; maskcol++){
        int pixP = (row + maskrow)*width + (col + maskcol); //maskrow - row + 1;
        int maskP = (maskrow+S)*filtsize + (maskcol+S);
        if((pixP < height*width) && (pixP >= 0)){
          sum += filter[maskP] * hData[pixP];
        }
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
}
