#include "mobilefacenet.h"

using namespace std;
using namespace cv;

Mat transpose_201(Mat *img)
{
	int i = 0, j = 0, k = 0;
	float *ptr = NULL;
	Mat ret_img = Mat::zeros(img->channels(), img->rows, CV_32FC(img->cols));

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			ptr = ret_img.ptr<float>(i, j);
			for (k = 0; k < ret_img.channels(); k++) {
				*ptr = *(img->ptr<float>(j, k) + i);
				ptr++;
			}
		}
	}

	return ret_img;
}

void normalize_m(Mat *img)
{
	int i = 0, j = 0, k = 0;
	float *ptr = NULL;

	for (i = 0; i < img->rows; i++) {
		for (j = 0; j < img->cols; j++) {
			ptr = img->ptr<float>(i, j);
			for (k = 0; k < img->channels(); k++) {
				*ptr = (*(ptr) - 127.5) * 0.0078125;
				ptr++;
			}
		}
	}

	return ;
}

Mat get_out(const char *filename, int *shape)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(shape[0], shape[1], CV_32FC1);

	FILE *f = fopen(filename, "rb+");
	float *tmp1 = (float*)malloc(shape[0] * shape[1] * sizeof(float));
	if (tmp1 == NULL)
		printf("malloc error\n");

	fread(tmp1, shape[0] * shape[1] * sizeof(float), 1, f);
	float *tmp2 = tmp1;

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*ret_img.ptr<float>(i, j) = *tmp2;
			tmp2++;
		}
	}

	fclose(f);
	free(tmp1);
	return ret_img;
}

Mat mat_add(Mat *img1, Mat *img2)
{
	int i = 0, j = 0;
	Mat ret_img = Mat::zeros(img1->rows, img1->cols, CV_32FC1);

	for (i = 0; i < ret_img.rows; i++) {
		for (j = 0; j < ret_img.cols; j++) {
			*ret_img.ptr<float>(i, j) = *img1->ptr<float>(i, j) + *(img2->ptr<float>(i, j));
		}
	}

	return ret_img;
}

void normalize_l2(Mat *img)
{
	int i = 0, j = 0;
	float total = 0;
	float square_root = 0;
	float *ptr = NULL;

	for (i = 0; i < img->rows; i++) {
		ptr = img->ptr<float>(i);
		for (j = 0; j < img->cols; j++) {
			total += pow(*ptr, 2);
			ptr++;
		}

		square_root = sqrt(total);
		ptr = img->ptr<float>(i);
		for (j = 0; j < img->cols; j++) {
			*ptr = *(ptr) / square_root;
			ptr++;
		}
		total = 0;
	}

	return ;
}

float get_distance(Mat *img, char *file)
{
	int i = 0, j = 0;
	float ret = 0;
	Mat cmp_img = Mat::zeros(img->rows, img->cols, CV_32FC1);

	FILE *f = fopen(file, "rb+");
	float *file_buf = (float*)malloc(img->rows * img->cols * sizeof(float));
	fread(file_buf, img->rows * img->cols * sizeof(float), 1, f);
	float *tmp = file_buf;

	for (i = 0; i < img->rows; i++) {
		for (j = 0; j < img->cols; j++) {
			*cmp_img.ptr<float>(i, j) = *tmp;
			tmp++;
		}
	}

	for (i = 0; i < img->rows; i++) {
		for (j = 0; j < img->cols; j++) {
			ret += pow((*img->ptr<float>(i, j) - *cmp_img.ptr<float>(i, j)), 2);
		}
	}

	return ret;
}
