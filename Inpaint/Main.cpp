#include "Inpaint.h"

int main()
{
	Mat Src = imread("cow_img.bmp");
	Mat Mask = imread("cow_mask.bmp", 0);

	if (Src.data == NULL || Mask.data == NULL) {
		cout << "No image data!" << endl;
	}

	imshow("src", Src);
	imshow("mask", Mask);

	Inpaint test(Src, Mask);
	test.BuildPyr();

	waitKey();

	return 0;
}