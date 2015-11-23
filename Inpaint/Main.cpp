#include "Inpaint.h"

int main()
{
	Mat Src = imread("E:\\Projects\\Algorithm\\TestOpencv\\Inpaint\\cow_img.bmp");
	Mat Mask = imread("E:\\Projects\\Algorithm\\TestOpencv\\Inpaint\\cow_mask.bmp", 0);

	if (Src.data == NULL || Mask.data == NULL) {
		cout << "No image data!" << endl;
	}

	imshow("src", Src);
	imshow("mask", Mask);

	Inpaint test(Src, Mask);
	test.Run();
	//test.PrintMaskValue();
	 
	waitKey();

	return 0;
}