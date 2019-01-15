#include "mobilefacenet.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	const char *out1_file = "out1.bin";
	const char *out2_file = "out2.bin";
	int out_shape[2] = {1, 128};
	float ret = 0;

	if (argc < 3) {
		printf("Usage: %s <detect image> <compare face>\n", argv[0]);
		return -1;
	}

	Mat img, img_float, img1, img2, out, out1, out2;

	img = imread(argv[1]);
	cvtColor(img, img, CV_RGB2BGR, img.channels());

	img_float;
	img.convertTo(img_float, CV_32FC3);

	img1 = transpose_201(&img_float);
	normalize_m(&img1);
	out1 = get_out(out1_file, out_shape);

	flip(img_float, img2, 1);
	img2 = transpose_201(&img2);
	normalize_m(&img2);
	out2 = get_out(out2_file, out_shape);

	out = mat_add(&out1, &out2);
	normalize_l2(&out);

	ret = get_distance(&out, argv[2]);
	printf("distance = %.8f\n", ret);

	return 0;
}
