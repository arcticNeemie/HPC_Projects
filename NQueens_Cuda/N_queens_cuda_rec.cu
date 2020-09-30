#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common/inc/helper_cuda.h"
#include "common/inc/helper_functions.h"
/*
  Function definitions
*/
void usage(char prog_name[]);
void initialize_board(int *board, int n);
void print_solution(int *board, int n);
// bool N_queens(int **board, int col, int n);
/*
Function which checks if it is safe to add a queen in this row and column
- This function does a row check, a col check and a diagonal check.
*/
__device__ bool check_is_safe(int *board, int row, int col, int n){
  // checking the row
  int i, j;
  for(i = 0; i < col; i++){
    if(board[row * n + i] == 1){
      return false;
    }
  }
  // checking upper diagonal, left side
  for(i = row, j = col; i >= 0 && j >= 0; i--, j--){
    if(board[i * n + j] == 1){
      return false;
    }
  }

  // checking lower diagonal, left side
  for(i = row, j = col; i < n && j >= 0; i++, j--){
    if(board[i * n + j] == 1){
      return false;
    }
  }


  /* we only check the left side in this function as the right side from col is
  not allocated yet.
  */
  return true;
}
/*
Nqueens recursive function with backtracking

*/

__device__ bool N_queens(int *board, int col, int n){

  // if there are N queens are placed return true
  if(col >= n){
    return true;
  }

  int thread = threadIdx.x + blockDim.x*blockIdx.x;
  // consider column and try placing a queen in all the rows one by one
  for(int i = 0; i < n; i++){
    // check if placement is safe
    if(check_is_safe(board, i, col, n)){
      board[i * n + col] = 1;
      if(N_queens(board, col+1, n)){
        return true;
      }
      // if placing queen at row i and col col, then remove the queen i.e backtrack
      board[i * n + col] = 0;

      // int *board_new = (int *)malloc(n*n*sizeof(int));
      // for(int k = 0; k < n; k++){
      //   for(int l = 0; l < n; l++){
      //     board_new[k * n + l] = board[k * n + l];
      //   }
      // }
    }
  }
  // if queen cannot be placed in any row in this column
  return false;
}
__global__ void find_solution(int *board_in, int n){

  int thread = threadIdx.x + blockDim.x*blockIdx.x;
  N_queens(board_in, thread, n);

}

int main(int argc, char *argv[]){

  int N; // size of N x N board
  int *board_d = NULL;
  int *board_h = NULL;
  int *board_out = NULL;
  int devID = findCudaDevice(argc, (const char **) argv);
  // int *board_d_out = NULL;

  if(argc != 2){
    usage(argv[0]);
    exit(-1);
  }
  N = atoi(argv[1]);

  // creating an N x N grid of threads
  int num_blocks = 1;
  int num_threads = N;
  int size = N*N*sizeof(int);
  // allocating memory for the host array
  board_h = (int *)malloc(N * N * sizeof(int));
  board_out = (int *)malloc(N * N * sizeof(int));
  checkCudaErrors(cudaDeviceSynchronize());

  StopWatchInterface *timer = NULL;

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  initialize_board(board_h, N);
  // print_solution(board_h, N);
  checkCudaErrors(cudaMalloc((void **) &board_d, size));
  checkCudaErrors(cudaMemcpy(board_d, board_h, size, cudaMemcpyHostToDevice));

  find_solution<<<num_blocks, num_threads>>>(board_d, N);
  getLastCudaError("Kernel Failed");
  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);

  checkCudaErrors(cudaMemcpy(board_out, board_d, size, cudaMemcpyDeviceToHost));
  // checking function here to ensure that the result that comes out is correct
  printf("\nSolution to placing %d queens on a %d x %d board is:\n\n", N, N, N);
  print_solution(board_out, N);
  printf("\nRunning time for placing %d queens on a %d x %d board %lf seconds\n", N, N, N, sdkGetTimerValue(&timer)/1000);

  free(board_h);
  free(board_out);
  cudaFree(board_d);
  return 0;
}
/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <size_of_board>\n", prog_name);
} /* usage */

// function which initializes the NxN board
void initialize_board(int *board, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j =0; j < n; j++){
      board[i * n + j] = 0;
    }
  }
}
// function which prints the result
void print_solution(int *board, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf(" %d ", board[i * n + j]);
    }
    printf("\n");
  }
}
