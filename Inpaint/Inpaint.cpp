#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat img = imread("cow_img.bmp");
	if (img.empty())
	{
		cout << "´ò¿ªÍ¼ÏñÊ§°Ü£¡" << endl;
		return -1;
	}
	namedWindow("image", CV_WINDOW_AUTOSIZE);
	imshow("image", img);
	waitKey();

	return 0;
}