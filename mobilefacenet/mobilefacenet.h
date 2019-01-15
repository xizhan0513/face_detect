#ifndef __MOBILEFACENET_H__
#define __MOBILEFACENET_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat transpose_201(Mat*);
void normalize_m(Mat*);
Mat get_out(const char*, int*);
Mat mat_add(Mat*, Mat*);
void normalize_l2(Mat*);
float get_distance(Mat*, char*);

#endif
