#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include "source.c"


// struct Point;  // forward declared for encapsulation
// Point* Point_create(int x, int y);  // equivalent to "new Point(x, y)"
// void Point_destroy(Point* self);  // equivalent to "delete point"
// void Point_reset(Point* self);
// int Point_x(Point* self);  // equivalent to "point->x()"
// int Point_y(Point* self);  // equivalent to "point->y()"
// void swap(int x, int y) {
//     int tmp = x;
//     x = y;
//     y = tmp;
// }
//
// int main() {
//     int m = 6;
//     int n = 10;
//     printf("Before swapping (m,n) evaluate to: (%d,%d)\n",m,n);
//     swap(m,n);
//     printf("After swapping (m,n) evaluate to: (%d,%d)\n",m,n); // m and n still equal to their original values
// }

// void swap(int* x, int* y) {
//     int tmp = *x; // the * operator is used to "de-reference" the pointer
//                   // i.e., to get the actual object which the pointer points to
//     *x = *y;
//     *y = tmp;
// }
// void printArray(double arr[], int size)
// {
//     // int s = sizeof(arr) / sizeof(int);
//     int i;
//     for (i=0; i < size; i++)
//         printf("%f\n", arr[i]);
//     // printf("n");
// }

int main() {

  // struct Point new;
  // struct Point *new = malloc(sizeof(struct Point));
  // new->n = 3;
  // for(int i = 0; i < 3; i++){
  //   int r = rand() % 100;
  //   new->dimension[i] = r;
  // }
  //
  // for(int i = 0; i < 9; i++){
  //   printf("i: %d\n %d\n", i, new->dimension[i]);

  // }

  int arr[10];
  for(int i = 0; i<11; i++){
    int r = rand() %10;
    arr[i] = r;
    printf("i:%d: %d\n", i, arr[i]);
  }









    // int m = 6;
    // int n = 10;
    // // printf("Before swapping (m,n) evaluate to: (%d,%d)\n",m,n);
    // swap(&m,&n); // now swap takes two integer pointers as arguments
    // // printf("After swapping (m,n) evaluate to: (%d,%d)\n",m,n); // m and n now have swapped their original values
    // double arr[] = {1,2,3,4};
    // double new[4];
    // for (int i = 0; i<4; i++){
    //   new[i] = arr[i];
    // }
    // printArray(new, 4);

}
