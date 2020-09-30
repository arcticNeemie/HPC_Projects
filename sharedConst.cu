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

#define filtsize 3
// #define filtsize 5
// #define filtsize 7
#define TILE_WIDTH 14 //For filtsize = 3
// #define TILE_WIDTH 10 //For filtsize = 5
// #define TILE_WIDTH 8 //For filtsize = 7
__device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
// __device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0,
//                                                            0.0,  0.0, 0.0, 0.0, 0.0};
// __device__ __constant__ float filter[filtsize*filtsize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//                                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//                                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//                                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
//                                                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

const char *imageFilename = "thanos.pgm";

//Function headers
__global__ void sharedConvolute(float* dData, float* hData, int height, int width);

int main(int argc, char **argv)
{
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //Load image
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

    float* dData = NULL;
    checkCudaErrors(cudaMalloc((void** )&dData, size));
    checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice)); //dData now contains the image

    float hEdge3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float hSharpen3[] = {-1.0, -1.0, -1.0, -1.0, 9, -1.0, -1.0, -1.0, -1.0};
    float hAverage3[] = {0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111};
    //
    float hSharpen5[] = {-1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, 25, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0};
    //
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
    //
    float hAverage7[] = {1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
                         1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49};


    int fsize =  filtsize*filtsize*sizeof(float);

    checkCudaErrors(cudaMemcpyToSymbol(filter, hEdge3, fsize));


    checkCudaErrors(cudaDeviceSynchronize());

    float* dOut = NULL;
    checkCudaErrors(cudaMalloc((void** )&dOut, size));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(width/dimBlock.y, height/dimBlock.x);

    StopWatchInterface *timer2 = NULL;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);
    sharedConvolute<<<dimGrid, dimBlock, 0>>>(dOut, dData, height, width);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer2);

    float t2 = sdkGetTimerValue(&timer2)/1000.0f;
    printf("Shared took %f time \n",t2);
    sdkDeleteTimer(&timer2);


    float* hOut = (float* )malloc(size);
    cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);

    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 5, "_SHARED_EDGE.pgm");
    sdkSavePGM(outputFilename, hOut, width, height);
    printf("Wrote '%s' to file\n", outputFilename);

    sdkStopTimer(&timer);
    float t = sdkGetTimerValue(&timer) / 1000.0f;
    printf("Shared overhead took %f time \n", t-t2);
    sdkDeleteTimer(&timer);

}

__global__ void sharedConvolute(float* dData, float* hData, int height, int width){

    __shared__ float shared_tile[16][16];

  int rowOut = threadIdx.y + blockIdx.y * 16;
  int colOut = threadIdx.x + blockIdx.x * 16; //Can only iterate through active part of tile as that is the output we set

  int rowT = rowOut;
  int colT = colOut;

  if ((rowT >= 0) && (rowT < height) && (colT >= 0) && (colT < width)){
    shared_tile[threadIdx.y][threadIdx.x] = hData[rowT * width + colT];
  }
  else{
    shared_tile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  if(rowOut<height && colOut<width){
    float sum = 0.0;
    if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH){
      for(int maskrow = 0; maskrow < filtsize; maskrow++){
        for(int maskcol = 0; maskcol < filtsize; maskcol++){
          int maskP = (maskrow)*filtsize + (maskcol);
            sum += filter[maskP] * shared_tile[maskrow+threadIdx.y][maskcol + threadIdx.x];
          }
        }
      }
      else if(threadIdx.y < width && threadIdx.x < height){
        for(int maskrow = 0; maskrow < filtsize; maskrow++){
          for(int maskcol = 0; maskcol < filtsize; maskcol++){
            int pixP = (threadIdx.y  + maskrow)*width + (threadIdx.x + maskcol); //maskrow - row + 1;
            int maskP = (maskrow)*filtsize + (maskcol);
              sum += filter[maskP] * hData[pixP];
            }
          }
      }
    __syncthreads();
    int pixPos = rowOut*width + colOut;
    dData[pixPos] = sum;
    if (dData[pixPos] < 0){
      dData[pixPos] = 0;
    }
    else if(dData[pixPos] > 1){
      dData[pixPos] = 1;
    }
  }
}
