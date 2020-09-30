#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <mpi.h>

#define MAXSIZE 35
#define PRINT


/*
  This function checks if the board is valid
*/
bool check(int *board, int size){
  int vals_pos[MAXSIZE] = {0};
  int vals_neg[MAXSIZE] = {0};

  for(int i = 0; i<size; i++){
    vals_neg[i] = i-board[i];
    vals_pos[i] = i+board[i];
  }

  for(int i = 0; i<size; i++){
    for(int j = 0; j<size; j++){
      if(i != j && ((vals_neg[i] == vals_neg[j]) || (vals_pos[i] == vals_pos[j]))){
        // Board is not valid
        return false;
      }
    }
  }
  //Board is valid
  return true;
}
/*
Takes in two pointers and swaps them
*/
void swap(int *x, int *y){
  int temp = *x;
  *x = *y;
  *y = temp;
}

/*
Calculates all valid permutations of queen placement
*/
void permute(int *board, int idx, int size, int *count){
  if(size == idx){
    if(check(board, size)){
      (*count)++;
      #ifdef PRINT
      for(int i = 0; i<size; i++){
        printf(" %d ", board[i]);
      }
      printf("\n");
      #endif
    }
    return;
  }
  int j = idx;
  for(j = idx; j<size; j++){
    swap(board+idx, board+j);
    permute(board, idx+1, size, count);
    swap(board+idx, board+j);
  }
  return;
}

/*
Main function
*/
int main(int argc, char *argv[]){
  int board[MAXSIZE];
  int num_procs;
  int rank;
  int solutions = 0;
  double _time;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(rank == 0){
    if(argc != 2){
      printf("Invalid usage\n");

      MPI_Finalize();
      exit(-1);
    }
  }

  int local_solution = 0;
  int size = atoi(argv[1]);

  for(int i = 0; i<size; i++){
    board[i] = i+1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  _time = -MPI_Wtime();

  for(int i = rank; i<size; i+=num_procs){
    swap(board+i, board);
    permute(board, 1, size, &local_solution);
    swap(board+i, board);
  }

  MPI_Reduce(&local_solution, &solutions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  _time += MPI_Wtime();

  if(rank == 0){
    printf("\n Total number of solutions: %d\n", solutions);
    printf("Time: %8.3f\n", _time*1000);
    fflush(stdout);
  }

  MPI_Finalize();
  return 0;


}
