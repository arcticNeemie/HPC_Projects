//Parallel code
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define NUM_THREADS 4

//n =  how many q points there arguments
//m = how many p points there are

struct Point{
  double dimension[128];  //d
  double distance[5000];  //size m of p points
};
void create_data(struct Point points[], int size, int d, int m){
  for(int i = 0; i < size; i++){
    for(int j = 0; j < d; j++){
      int s = m*2;
      points[i].dimension[j] = rand() %s;
    }
  }
}
double Euclidean_serial(struct Point p, struct Point q, int d){
  double sum = 0;
  #pragma omp parallel
{
  #pragma omp for
  for(int i = 0; i<d; i++){
    sum += pow((q.dimension[i] - p.dimension[i]), 2);
  }
}
  return sqrt(sum);
}

double Manhattan_serial(struct Point p, struct Point q, int d){
  double sum = 0;
  #pragma omp parallel
  {
    #pragma omp for
  for(int i = 0; i<d; i++){
    double s = abs(q.dimension[i] - p.dimension[i]);
    sum = sum + s;
  }
}
  return sum;

}
void swap(double *xp, double *yp)
{
  double temp = *xp;
  *xp = *yp;
  *yp = temp;
}

void Bubble_serial(struct Point *q, int m){
  #pragma omp parallel
  {
  #pragma omp for collapse(2)
    for(int i = 0; i < m; i++){
      for(int j = 0; j < m-i-1; j++){
        if(q->distance[j] > q->distance[j+1]){
          swap(&q->distance[j], &q->distance[j+1]);
        }
      }
    }
  }
}
int partition (struct Point **q, int *low, int *high){
    double pivot = (*q)->distance[*high];
    int i = (*low - 1);  // Index of smaller element

    for (int j = *low; j <= *high- 1; j++){
        if ((*q)->distance[j] <= pivot){
            i++;    // increment index of smaller element
        swap(&(*q)->distance[i], &(*q)->distance[j]);
        }
      }
          swap(&(*q)->distance[i+1], &(*q)->distance[*high]);
    return (i + 1);
}
void quickSort_serial(struct Point *q, int low, int high)
{
if(low < high && high - low <= 1000)
{
  int pi = partition(&q, &low, &high);
        quickSort_serial(&(*q), low, pi - 1);
        quickSort_serial(&(*q), pi + 1, high);
}
if(low < high && high - low >1000){
      int pi = partition(&q, &low, &high);
      #pragma omp parallel
      {
        #pragma omp taskgroup
        {
          #pragma omp task
          quickSort_serial(&(*q), low, pi - 1);
          #pragma omp task
          quickSort_serial(&(*q), pi + 1, high);
        }
      }
  }
}
void insertionSort_serial(struct Point *q, int m)
{
  struct Point key;
   int j, i;
   #pragma omp parallel
   {
     #pragma omg for
     for (i = 1; i < m; i++)
   {
       key = *q;
       double dkey = key.distance[i];
       j = i-1;
       while (j >= 0 && q->distance[j] > dkey)
       {
           q->distance[j+1] = q->distance[j];
           j = j-1;
       }
       q->distance[j+1] = dkey;
   }
 }
}

void KNN_serial(struct Point p[], struct Point q[], int n, int m, int d, double timer[2]){
  double start_time, run_time = 0;
  double start_time2, run_time2 = 0;
  start_time = omp_get_wtime();
  #pragma omp parallel
  {
  #pragma omp for collapse(2)
  for(int i = 0; i<n; i++){
    for(int j = 0; j<m; j++){
      q[i].distance[j] = Euclidean_serial(p[j], q[i], d);
      // q[i].distance[j][1] = Manhattan_serial(p[j], q[i], d);
    }
  }
}
run_time += omp_get_wtime() - start_time;
timer[0] = run_time;

start_time2 = omp_get_wtime();
  #pragma omp parallel
  {
  #pragma omp for
    for(int i = 0; i<n; i++){
    // Bubble_serial(&q[i], m);
    // insertionSort_serial(&q[i],m);
    quickSort_serial(&q[i], 0, m-1);
  }
}

// printf("Manhattan distance\n");
printf("Euclidean distance\n");
run_time2 += omp_get_wtime() - start_time2;
timer[1] = run_time2;
// printf("Bubble sort\n");
// printf("Insertion sort\n");
printf("Quick sort\n");
}

int main(){
  double timer[2];
  int m = 5000; //Number of p points
  int d = 128; //Number of dimensions
  int n = 1600; //Number of q points
  struct Point *p = (struct Point*)malloc(m * sizeof(struct Point));
  struct Point *q = (struct Point*)malloc(n * sizeof(struct Point));

  create_data(p, m, d, m);
  create_data(q, n, d, m);

  KNN_serial(p, q, n, m, d, timer);

  free(p);
  free(q);
  double t = timer[0] + timer[1];
  double t2 = (timer[0] / t)*100;
  double t3 = (timer[1] / t)*100;;
  printf("Reference m: %d\n Query n: %d\n Dimensions d: %d\n", m, n, d);
  printf("Distance: %f\n", t2);
  printf("Sorting: %f\n", t3);
  printf("Overall time: %f seconds\n", t);

}
