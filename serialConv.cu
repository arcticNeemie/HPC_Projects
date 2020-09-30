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

const char *imageFilename = "lena_bw.pgm";

//Function headers
// void WriteToFile(const char* imagePath, float* hData, int width, int height);
void convolute(float* outData, float*hData, int height, int width, float* mask, int masksize);
__global__ void convoluteGPU(float* dData, float* hData, int height, int width, float* mask, int masksize);


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

  float *outData = (float*) malloc(size);
  float* dData = NULL;
  float* dOut = NULL;
  checkCudaErrors(cudaMalloc((void** )&dData, size));
  checkCudaErrors(cudaMalloc((void** )&dOut, size));
  checkCudaErrors(cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice));

  //Define mask
  int masksize = 3;
  float* hSharpen = (float*) malloc(masksize*masksize*sizeof(float));
  float* hEdge = (float*) malloc(masksize*masksize*sizeof(float));
  float* hAverage = (float*) malloc(masksize*masksize*sizeof(float));

  hEdge[0] = -1.0;
  hEdge[1] = 0.0;
  hEdge[2] = 1.0;
  hEdge[3] = -2.0;
  hEdge[4] = 0.0;
  hEdge[5] = 2.0;
  hEdge[6] = -1.0;
  hEdge[7] = 0.0;
  hEdge[8] = 1.0;

  hSharpen[0] = -1.0;
  hSharpen[1] = -1.0;
  hSharpen[2] = -1.0;
  hSharpen[3] = -1.0;
  hSharpen[4] = 9.0;
  hSharpen[5] = -1.0;
  hSharpen[6] = -1.0;
  hSharpen[7] = -1.0;
  hSharpen[8] = -1.0;

  hAverage[0] = 0.111;
  hAverage[1] = 0.111;
  hAverage[2] = 0.111;
  hAverage[3] = 0.111;
  hAverage[4] = 0.111;
  hAverage[5] = 0.111;
  hAverage[6] = 0.111;
  hAverage[7] = 0.111;
  hAverage[8] = 0.111;

  float* dFilter = NULL;
  int fsize =  masksize*masksize*sizeof(float);
  checkCudaErrors(cudaMalloc((void** )&dFilter,fsize));
  checkCudaErrors(cudaMemcpy(dFilter, hSharpen, fsize, cudaMemcpyHostToDevice));
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(height/dimBlock.x, width/dimBlock.y, 1);
  // checkCudaErrors(cudaDeviceSynchronize());
  // convoluteGPU<<<dimGrid, dimBlock>>>(dOut, dData, height, width, dFilter, fsize);
  // checkCudaErrors(cudaDeviceSynchronize());
  float* hOut = (float* )malloc(size);
  cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);
  // free(hOut);
  // free(dOut);
  // for(int i = 0; i<masksize*masksize; i++){
  //   printf("%f\n", hEdge[i]);
  // }

  convolute(outData, hData, height, width, hEdge, masksize);
  // for(int i = 0; i< height*width; i++){
  //   printf("%f", outDataEdge[i]);
  // }
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_SerialEdge.pgm");
  sdkSavePGM(outputFilename, hOut, width, height);
  printf("Wrote '%s' to file\n", outputFilename);

}

void convolute(float* dData, float* hData, int height, int width, float* mask, int masksize){
//row and col refer to image

int S = (masksize-1)/2; //Basically 1
  for (int row = 0; row < height; row++){
    for(int col = 0; col < width; col++){
      float sum = 0;
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

__global__ void convoluteGPU(float* dData, float* hData, int height, int width, float* mask, int masksize){
   int row = threadIdx.x + blockIdx.x * blockDim.x;
   int col = threadIdx.y + blockIdx.y * blockDim.y;

  int S = (masksize-1)/2;
  float sum = 0;
  int pixPos = row*width + col;
  dData[pixPos] = 0.0;
if(row<height && col<width){
  for(int maskrow = -S; maskrow <= S; maskrow++){
    for(int maskcol = -S; maskcol <= S; maskcol++){
      int pixP = (row + maskrow)*width + (col + maskcol); //maskrow - row + 1;
      int maskP = (maskrow+S)*masksize + (maskcol+S);
      if(pixP < height*width && pixP > 0 && maskP<masksize*masksize){
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
