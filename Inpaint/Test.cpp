#include "Inpaint.h"

void Inpaint::PrintMaskValue()
{
	for (int i = 0; i < maskImg[0].rows; i++)
	{
		for (int j = 0; j < maskImg[0].cols; j++)
		{
			cout << (int) maskImg[0].at<uchar>(i, j) << ' ';
		}
		cout << endl;
	}
}