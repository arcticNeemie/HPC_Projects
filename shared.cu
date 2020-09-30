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

#define TILE_WIDTH 14


// #define w (TILE_WIDTH + filtsize - 1) //size of tile after padding was added i.e. 20 for 3x3 filter


const char *imageFilename = "lena_bw.pgm";

//Function headers
__global__ void sharedConvolute(float* dData, float* hData, int height, int width, float* filter, int masksize);
// void callConst(float* hData, float* mask, int width, int height, unsigned int size, int masksize, char* imagePath);

int main(int argc, char **argv)
{
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
  // printf("%d, %d\n", height,width);

  unsigned int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // float* hSharpen = (float*) malloc(filtsize*filtsize*sizeof(float));
    // float* hEdge = (float*) malloc(filtsize*filtsize*sizeof(float));
    // float* hAverage = (float*) malloc(filtsize*filtsize*sizeof(float));

    float* dData = NULL;
    checkCudaErrors(cudaMalloc((void** )&dData, size));
    checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice)); //dData now contains the image

    int masksize = 3;
    // int masksize = 5;
    // int masksize = 7;

    float hEdge3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    // float hSharpen3[] = {-1.0, -1.0, -1.0, -1.0, 9, -1.0, -1.0, -1.0, -1.0};
    // float hAverage3[] = {0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111};
    //
    // float hSharpen5[] = {-1.0, -1.0, -1.0, -1.0, -1.0,
    //                     -1.0, -1.0, -1.0, -1.0, -1.0,
    //                     -1.0, -1.0, 25, -1.0, -1.0,
    //                     -1.0, -1.0, -1.0, -1.0, -1.0,
    //                     -1.0, -1.0, -1.0, -1.0, -1.0};

    // float hAverage5[] = {0.04, 0.04, 0.04, 0.04, 0.04,
    //                      0.04, 0.04, 0.04, 0.04, 0.04,
    //                      0.04, 0.04, 0.04, 0.04, 0.04,
    //                      0.04, 0.04, 0.04, 0.04, 0.04,
    //                      0.04, 0.04, 0.04, 0.04, 0.04,};
    //
    // float hSharpen7[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, 49, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0,
    //                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 ,-1.0};
    //
    // float hAverage7[] = {1.0/49, 1.0/49, 1.0/7, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49,
    //                     1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49, 1.0/49};

    // float* dFilter = NULL;
    float* dFilter = NULL;
    int fsize =  masksize*masksize*sizeof(float);
    checkCudaErrors(cudaMalloc((void** )&dFilter,fsize));
    checkCudaErrors(cudaMemcpy(dFilter, hEdge3, fsize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMalloc((void** )&dFilter,fsize));
    // checkCudaErrors(cudaMemcpy(dFilter, hSharpen, fsize, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpyToSymbol(filter, hSharpen7, fsize));


    checkCudaErrors(cudaDeviceSynchronize());

    float* dOut = NULL;
    checkCudaErrors(cudaMalloc((void** )&dOut, size));

    dim3 dimBlock(16, 16);
    // dim3 dimGrid(width-1/TILE_WIDTH+1, height-1/TILE_WIDTH+1, 1);
    dim3 dimGrid(width/dimBlock.y, height/dimBlock.x);
    // checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    sharedConvolute<<<dimGrid, dimBlock, 0>>>(dOut, dData, height, width, hEdge3, masksize);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    // printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    // printf("%.2f Mpixels/sec\n",
           // (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    float time = sdkGetTimerValue(&timer);
    printf("Constant memory took %f  \n", time/1000.0f);
    sdkDeleteTimer(&timer);
    // float time = sdkGetTimerValue(&timer) / 1000.0f;
    // printf("Shared memory took %f (ms) \n", time);
    // sdkDeleteTimer(&timer);

    float* hOut = (float* )malloc(size);
    cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);

    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 6, "_sharedEdge.pgm");
    sdkSavePGM(outputFilename, hOut, width, height);
    printf("Wrote '%s' to file\n", outputFilename);

}

__global__ void sharedConvolute(float* dData, float* hData, int height, int width, float* filter, int filtsize){
  // int filtsize = 3;
  // int S = (filtsize-1)/2;
  // const int w = TILE_WIDTH + (2*S);
  // int w = 16;
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
  // dData[pixPos] = 0.0;
  if(rowOut<height && colOut<width){
    float sum = 0.0;
    if(threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH){  //maskP<filtsize*filtsize
      for(int maskrow = 0; maskrow < filtsize; maskrow++){
        for(int maskcol = 0; maskcol < filtsize; maskcol++){
          // int pixP = (rowT + maskrow)*width + (colT + maskcol); //maskrow - row + 1;
          int maskP = (maskrow)*filtsize + (maskcol);
            sum += filter[maskP] * shared_tile[maskrow+threadIdx.y][maskcol + threadIdx.x];
          }
        }
      }
      else if(threadIdx.y < width && threadIdx.x < height){
        for(int maskrow = 0; maskrow < filtsize; maskrow++){
          for(int maskcol = 0; maskcol < filtsize; maskcol++){
            int pixP = (rowT + maskrow)*width + (colT + maskcol); //maskrow - row + 1;
            int maskP = (maskrow)*filtsize + (maskcol);
              sum += filter[maskP] * hData[pixP];
            }
          }
      }
    __syncthreads();
    int pixPos = rowOut*width + colOut; //row*width+col
    dData[pixPos] = sum;
    if (dData[pixPos] < 0){
      dData[pixPos] = 0;
    }
    else if(dData[pixPos] > 1){
      dData[pixPos] = 1;
    }
  }
}
