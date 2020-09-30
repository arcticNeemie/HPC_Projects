#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>
/*
  Unncomment line 9 below to print solution
*/
// #define PRINT

// global solution count
int total_solutions = 0;

// checks if queens attack each other
bool check_attack(int board[], int col){
  for(int i = 0; i < col; i++){
    /*
      This checks to see if there is an already placed queen which blocks
      the current queen being placed. We only check the row position here since
      we add queens column wise.
    */
    if(board[i] == board[col]){
      return false;
    }
    /*
      This checks to see if the queen being placed is in diagonal conflict
      with any other queen already on the board.
    */
    if((abs(board[i] - board[col])) == (col - i)){
      return false;
    }
  }
  return true;
}
// initializes the board to 0's
void initialize_board(int *solution, int n){
  int i, j;
  for(i = 0; i < n; i++){
    for(j =0; j < n; j++){
      solution[i * n + j] = 0;
    }
  }
}
// function which prints the result
void print_solution(int *solution, int row_board[], int n){
  int i, j;
  // fill in the board
  for(j = 0; j < n; j++){
    // 1D solution into 2D
    solution[row_board[j] * n + j] = 1;
  }
  // prints the board
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      printf(" %d ", solution[i * n + j]);
    }
    printf("\n");
  }
}
void n_queens(int board[], int col, int n, int queen){
  int i;

  if(col == n){ // we have reached a solution
    total_solutions++;
    // printing of solution
    #ifdef PRINT
    // allocating memory for the array
    int *solution = (int *)malloc(n * n * sizeof(int));
    initialize_board(solution, n);
    // here we want to print out the solution in 2D board format
    print_solution(solution, board, n);
    printf("\n");
    #endif
  }else{
    board[0] = queen;
    for(i = 0; i < n; i++){
      board[col] = i;
      if(check_attack(board, col)){
        // here we do the recursion, if the position is valid
        n_queens(board, col + 1, n, queen);
      }
    }
  }
}

/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage: %s <size_of_board>\n", prog_name);
} /* usage */

/*
  Main
*/
int main(int argc, char *argv[]){

  int rank, num_procs;
  int N;  // size of the board
  double start, end, time;

  if(argc != 2){
    usage(argv[0]);
    exit(-1);
  }
  N = atoi(argv[1]);
  int board[N];
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  if(rank != 0){
    /*
      Starting the solution at col 1 since we set col 0 in the array
      to the corressponding rank (process). This placement of the queen is valid
      as it is the first queen placed. We set the position of the queen to rank - 1
      since rank 0 does no computation and we only want to set the value of col 0
      from 1 to N, as number of processors = N+1.
    */
    n_queens(board, 1, N, rank-1);
    // sending the total solution count that the current rank has computed to rank 0.
    MPI_Send(&total_solutions, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }else{
    start = MPI_Wtime();
    // iterate through each process not including rank 0
    for(int source = 0; source < num_procs - 1; source++){
      int rank_answer;
      // receive each rank's solution count
      MPI_Recv(&rank_answer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // sum each ranks solution count
      total_solutions += rank_answer;
    }
    end = MPI_Wtime();
    time = end - start;
    printf("\nTotal number of solutions: %d\n", total_solutions);
    printf("\nRunning time for placing %d queens on a %d x %d board %lf seconds\n", N, N, N, time);
  }
  MPI_Finalize();
  return 0;
}
