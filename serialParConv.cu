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

const char *imageFilename = "thanos.pgm";

// #define masksize 3
//Function headers

void convolute(float* outData, float*hData, int height, int width, float* mask, int masksize);
__global__ void convoluteNaive(float* dData, float* hData, int height, int width, float* mask, int masksize);
void callSerial(float* hData, float* mask, int width, int height,  unsigned int size, int masksize, char* imagePath);
void callNaive(float* hData, float* mask, int width, int height,  unsigned int size, char* imagePath, int masksize);
void printSpecs();



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

  int size = width * height * sizeof(float);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    int masksize = 3;
    // int masksize = 5;
    // int masksize = 7;

    float hEdge3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};
    float hSharpen3[] = {-1.0, -1.0, -1.0, -1.0, 9, -1.0, -1.0, -1.0, -1.0};
    float hAverage3[] = {0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111};
    //
    float hSharpen5[] = {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,-1.0, -1.0, 25, -1.0, -1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

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

    callSerial(hData,hSharpen3,width, height, size, masksize, imagePath);
    // callNaive(hData, hAverage7, width, height, size, imagePath, masksize);
        printSpecs();
}

///////////////////////////////////////
///////////// CONVOLUTIONS ///////////
/////////////////////////////////////
void convolute(float* dData, float* hData, int height, int width, float* mask, int masksize){
//row and col refer to image
int S = (masksize-1)/2;
  for (int row = 0; row < height; row++){
    for(int col = 0; col < width; col++){
      float sum = 0.0;
      int pixPos = row*width + col;
      dData[pixPos] = 0.0;

        for(int maskrow = -S; maskrow <= S; maskrow++){
          for(int maskcol = -S; maskcol <= S; maskcol++){
          //Indices:
            int pixP = (row + maskrow)*width + (col + maskcol); //maskrow - row + 1;
            int maskP = (maskrow+S)*masksize + (maskcol+S);
          //To avoid padding, make sure the mask is inside the image
            if(pixP < height*width && pixP > 0){
              sum += mask[maskP] * hData[pixP];
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
}

__global__ void convoluteNaive(float* dData, float* hData, int height, int width, float* mask, int masksize){ //dOut, dData
 unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
 unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

 int S = (masksize-1)/2;
 float sum = 0.0;
 int pixPos = row*width + col;
 if(row<height && col<width){
   for(int maskrow = -S; maskrow <= S; maskrow++){
     for(int maskcol = -S; maskcol <= S; maskcol++){
       int pixP = (row + maskrow)*width + (col + maskcol); //maskrow - row + 1;
       int maskP = (maskrow+S)*masksize + (maskcol+S);
       if((pixP < height*width) && (pixP >= 0)){
         sum += mask[maskP] * hData[pixP];
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
///////////////////////////////////////
//// FUNCTIONS TO CALL CONVOLUTIONs ///
/////////////////////////////////////
void callSerial(float* hData, float* mask, int width, int height, unsigned int size, int masksize, char* imagePath){
  float *dData = (float*) malloc(size); //Output

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  convolute(dData,hData,height,width,mask,masksize); //Function
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer) / 1000.0f;
  printf("Serial took %f s \n", time);
  sdkDeleteTimer(&timer);

  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 6, "_SerialAverage7.pgm");
  sdkSavePGM(outputFilename, dData, width, height);
  printf("Wrote '%s' to file\n", outputFilename);

  free(dData);
}
void callNaive(float* hData, float* mask, int width, int height,  unsigned int size, char* imagePath, int masksize){
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  float* dData = NULL;
  checkCudaErrors(cudaMalloc((void** )&dData, size));
  checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice));
  float* dFilter = NULL;
  int fsize =  masksize*masksize*sizeof(float);
  checkCudaErrors(cudaMalloc((void** )&dFilter,fsize));
  checkCudaErrors(cudaMemcpy(dFilter, mask, fsize, cudaMemcpyHostToDevice));
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width/dimBlock.x, height/dimBlock.y, 1);
  checkCudaErrors(cudaDeviceSynchronize());

  float* dOut = NULL;
  checkCudaErrors(cudaMalloc((void** )&dOut, size));

    StopWatchInterface *timer2 = NULL;
    sdkCreateTimer(&timer2);
    sdkStartTimer(&timer2);
    convoluteNaive<<<dimGrid, dimBlock>>>(dOut, dData, height, width, dFilter, masksize);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer2);
    float t2 = sdkGetTimerValue(&timer2) / 1000.0f;
    printf("Naive took %f time \n", t2);
    sdkDeleteTimer(&timer2);


    float* hOut = (float* )malloc(size);
    cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);

    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_NaiveAverage7.pgm");
    sdkSavePGM(outputFilename, hOut, width, height);
    printf("Wrote '%s' to file\n", outputFilename);
    sdkStopTimer(&timer);
    float t = sdkGetTimerValue(&timer) / 1000.0f;
    printf("Naive parallel overhead took %f time \n", t-t2);
    sdkDeleteTimer(&timer);


}

void printSpecs(){
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for(int i = 0; i< nDevices; i++){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Memory clock rate (Khz): %d\n",
    prop.memoryClockRate);
    printf("Memory bus width (bits): %d\n", prop.memoryBusWidth);
    printf("peak memory bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
