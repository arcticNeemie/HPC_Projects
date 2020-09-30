// Source file
struct Point {
  int dimension[3];
  double distance[2];  //size m
};

// // Constructor (without allocation)
// void Point_init(struct Point* self, double distance[], double dimension[]) {
//   self->distance[] = distance[];
//   self->dimension[] = dimension[];
//  }
//
// // Allocation + initialization (equivalent to "new Point(x, y)")
// struct Point* Point_create(double distance[], double dimension[]) {
//    struct Point* result =  (struct Point*)malloc(sizeof(struct Point));
//    Point_init(result, distance, dimension);
//    return result;
// }
//
// // Destructor (without deallocation)
// void Point_reset(struct Point* self) {
// }
//
// // Destructor + deallocation (equivalent to "delete point")
// void Point_destroy(struct Point* point) {
//   if (point) {
//      Point_reset(point);
//      free(point);
//   }
// }
//
// // // Equivalent to "Point::x()" in C++ version
// // int Point_x(Point* self) {
// //    return self->x;
// // }
// //
// // // Equivalent to "Point::y()" in C++ version
// // int Point_y(Point* point) {
// //    return self->y;
// // }
