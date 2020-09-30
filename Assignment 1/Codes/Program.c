//Serial code

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
// #include "source.c"


struct Point{
  double dimension[128];
  double distance[256][256];  //size m of p points
};
//n =  how many q points there arguments
//m = how many p points there are
void create_data(struct Point points[], int size, int d, int m){
  for(int i = 0; i < size; i++){
    for(int j = 0; j < d; j++){
      points[i].dimension[j] = rand() %50;
    }
  }
  for(int p = 0; p < size; p++){
    for(int k = 0; k < m; k++){
      points[p].distance[k][0] = k;
    }
  }
}

void printArray(int arr[], int size)
{
    // int s = sizeof(arr) / sizeof(int);
    int i;
    for (i=0; i < size; i++)
        printf("%d ", arr[i]);
    // printf("n");
}
double Euclidean_serial(struct Point p, struct Point q, int d){
  double sum = 0;
  for(int i = 0; i<d; i++){
    double s = (q.dimension[i] - p.dimension[i]) * (q.dimension[i] - p.dimension[i]);
    sum = sum + s;
  }
  double ED = sqrt(sum);
  return ED;
}

double Manhattan_serial(struct Point p, struct Point q, int d){
  double sum = 0;
  for(int i = 0; i<d; i++){
    double s = abs(q.dimension[i] - p.dimension[i]);
    sum = sum + s;
  }
  return sum;

}
void swap(double *xp, double *yp)
{
  double temp = *xp;
  *xp = *yp;
  *yp = temp;
}
// void swapD(double arr[][]){
//   double temp = arr[0][1];
//   arr[0][1] = arr[1][1];
//    arr[1][1] = temp;
// }
void Bubble_serial(struct Point *q, int m){   //Takes in q array
//     distarr[k] = q[l].distance[k];
    for(int i = 0; i < m; i++){
      for(int j = 0; j < m-i-1; j++){
        if(q->distance[j][1] > q->distance[j+1][1]){
//           // printf("fist %f\n\n", q[1].distance[1]);
          // printf("To swap:  %f and %f\n",q->distance[j] , q->distance[j+1]);
          swap(&q->distance[j][1], &q->distance[j+1][1]);
          swap(&q->distance[j][0], &q->distance[j+1][0]);
          // printf("After swap: %f and %f\n", q->distance[j] , q->distance[j+1]);
//           // double temp = points[i].distance[j];
//           // points[i].distance[j] = points[i].distance[j+1];
//           // points[i].distance[j+1] = temp;
        }
      }
    }
  }
int partition (struct Point **q, int *low, int *high)
{
    // double *pivot = arr[*high].distance;    // pivot
    double pivot = (*q)->distance[*high][1];
    int i = (*low - 1);  // Index of smaller element

    for (int j = *low; j <= *high- 1; j++){
            // If current element is smaller than or equal to pivot
        if ((*q)->distance[j][1] <= pivot){
            i++;    // increment index of smaller element
            swap(&(*q)->distance[i][1], &(*q)->distance[j][1]);
            swap(&(*q)->distance[i][0], &(*q)->distance[j][0]);
            // struct Point temp = arr[i];
            // arr[i] = arr[j];
            // arr[j] = temp;
        }
    }
    swap(&(*q)->distance[i+1][1], &(*q)->distance[*high][1]);
    swap(&(*q)->distance[i+1][0], &(*q)->distance[*high][0]);
    // struct Point t = arr[i+1];
    // arr[i+1] = arr[*high];
    // arr[*high] = t;
    return (i + 1);
}
//
// /*low  --> Starting index,
//   high  --> Ending index */
//
void quickSort_serial(struct Point *q, int low, int high)
{
    if (low < high)
    {
        // pi is partitioning index, arr[p] is now at right place
        int pi = partition(&q, &low, &high);

        // Separately sort elements before partition and after partition
        quickSort_serial(&(*q), low, pi - 1);
        quickSort_serial(&(*q), pi + 1, high);
    }
}
void insertionSort_serial(struct Point *q, int m)
{
  struct Point key;
   int j, i;
   for (i = 1; i < m; i++)
   {
       key = *q;
       double dkey = key.distance[i][1];
       j = i-1;

        //Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position
       while (j >= 0 && q->distance[j][1] > dkey)
       {
           q->distance[j+1][1] = q->distance[j][1];
           q->distance[j+1][0] = q->distance[j][0];
           j = j-1;
       }
       q->distance[j+1][1] = dkey; //should be key
       q->distance[j+1][0] = key.distance[i][0];
   }
}

void KNN_serial(struct Point p[], struct Point q[], int n, int m, int d){ //n = q(3); m = p(2)
  for(int i = 0; i<n; i++){
    for(int j = 0; j<m; j++){
      q[i].distance[j][1] = Euclidean_serial(p[j], q[i], d);
      // q[i].distance[j][1] = Manhattan_serial(p[j], q[i], d);
    }
  }
  // for(int i = 0; i<n; i++){
  //   printf("Old point: %d \n", i);
  //   for(int j = 0; j<m; j++){
  //     printf( "%f, %f\n",q[i].distance[j][0], q[i].distance[j][1]);
  //   }
  // }
for(int i = 0; i<n; i++){
    // Bubble_serial(&q[i], m);
    // insertionSort_serial(&q[i],m);
    quickSort_serial(&q[i], 0, m-1);
}


// printf("Manhattan distance\n");
printf("Euclidean distance\n");

// printf("Bubble sort\n");
// printf("Insertion sort\n");
printf("Quick sort\n");
// }
}

int main(){
  double start_time, run_time=0;
  // clock_t start, end;
  // double cpu_time_used;
  // *******
  int m = 256; //Number of p points
  int d = 128; //Number of dimensions
  int n = 1600; //Number of q points
  // struct Point p[m];
  // struct Point q[n];
  struct Point *p = (struct Point*)malloc(m * sizeof(struct Point));
  struct Point *q = (struct Point*)malloc(n * sizeof(struct Point));

  create_data(p, m, d, m);
  create_data(q, n, d, m);
  // int k = 3; //Number of nearest neighbours
  // for(int i = 0; i < n; i++){
  //   printf("Q point %d :\n", i);
  //   for(int j = 0; j < d; j++){
  //     printf("%f,\n", q[i].dimension[j]);
  //   }
  // }

  // for(int i = 0; i < m; i++){
  //   printf("P point %d :\n", i);
  //   for(int j = 0; j < d; j++){
  //     printf("%f,\n", p[i].dimension[j]);
  //   }
  // }
  start_time = omp_get_wtime();
  // start = clock();
  KNN_serial(p, q, n, m, d);
  // end = clock();
  run_time += omp_get_wtime() - start_time;
  // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Reference m: %d\n Query n: %d\n Dimensions d: %d\n Time: %f seconds\n", m, n, d, run_time);
  // for(int i = 0; i<n; i++){
  //   // printf("Old point: %d \n", i);
  //   for(int j = 0; j<m; j++){
  //     printf( "%f", q[i].distance[j][1]);
  //     printf(" ");
  //   }
  //   printf("\n");
  // }
  free(q);
  free(p);



// *******

//   for(int i = 0; i < n; i++){
//     double xs = rand() %100;
//     double ys = rand() %100;
//     int c = rand() %2;
//     arr[i].x = xs;
//     arr[i].y = ys;
//     arr[i].class = c;
//   }

// int m  = 32;
// int n = 200;
// struct Point p[m];  //Ref points i.e. m
// struct Point q[n];  //Query points i.e. n
// int d = 32;      //Dimension
// create_data(p, m);
// create_data(q,n);
//




  /*JUST FOR TESTING STUFF: */
  // int n = 2;
  // Point arr[2];
//   struct Point q1;
//   q1.dimension[0] = 4;
//   q1.dimension[1] = 7;
//   q1.dimension[2] = 2;
//   q1.distance[0][0] = 0;
//   q1.distance[1][0] = 1;
//
//   struct Point q2;
//   q2.dimension[0] = 3;
//   q2.dimension[1] = 2;
//   q2.dimension[2] = 1;
//   q2.distance[0][0] = 0;
//   q2.distance[1][0] = 1;
//
//   struct Point q3;
//   q3.dimension[0] = 7;
//   q3.dimension[1] = 3;
//   q3.dimension[2] = 4;
//   q3.distance[0][0] = 0;
//   q3.distance[1][0] = 1;
//
// struct Point p1;
//   p1.dimension[0] = 3;
//   p1.dimension[1] = 11;
//   p1.dimension[2] = 10;
//
//   struct Point p2;
//   p2.dimension[0] = 4;
//   p2.dimension[1] = 2;
//   p2.dimension[2] = 1;
//
//   struct Point p[2];
//   struct Point q[3];
//   int m = 2;
//   int n = 3;
//   int d = 3;
//   int k = 3;
//
//   p[0] = p1;
//   p[1] = p2;
//
//   q[0] = q1;
//   q[1] = q2;
//   q[2] = q3;

  // for(int i = 0; i<n; i++){
  //   for(int j = 0; j<m; j++){
  //     printf( "%f, %f\n",q[i].distance[j][0], q[i].distance[j][1]);
  //   }
  // }
   // KNN_serial(p, q, n, m, d);


  // double ED = Euclidean(p,q,3);
  // printf("ED is: %f\n", ED);


  // arr[3].x = 3;
  // arr[3].y = 2;
  //
  // // int ans = KNN_serial(arr, q, 4, k);
  //
  // int k = 3;
  // Point q;
  // q.x = 3;
  // q.y = 4;
  //  KNN_serial(arr, q, n, k);



  // double x1 = 4;
  // double x2 = 3;
  // double y1 = 7;
  // double y2 = 10;
  // double ans = Manhattan_serial(x1, x2, y1, y2);
  // int arr[] = {900,29,2,5,78,34,2,7,28};
  // int points[50];
  // int n = 50;
  // create_data(points, n);
  // insertionSort_serial(points,n);
  // Bubble_serial(arr,n);
  // quickSort_serial(arr,0,n-1);
  // printArray(arr,n);
  // printf("%f", ans);
}
