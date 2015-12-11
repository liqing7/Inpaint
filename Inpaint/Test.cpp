#include "Inpaint.h"

void Inpaint::PrintMaskValue()
{
	/*for (int i = 0; i < maskImg[6].rows; i++)
	{
		for (int j = 0; j < maskImg[6].cols; j++)
		{
			cout << (int) maskImg[6].at<uchar>(i, j) << ' ';
		}
		cout << endl;
	}*/

	for (int i = 0; i < 7; i++)
	{
		cout << "Mask " << i << endl;
		//cout << maskImg[i] << endl;
	}
}

void Inpaint::PrintMaskValue(const Mat& mask)
{
	cout << mask << endl;
}

void Inpaint::PrintOffsetMap(const Mat& offset)
{
	cout << offset << endl;
}

void Inpaint::PrintMaskValue(const Mask& mask)
{
	for (int i = 0; i < mask.row; i++)
	{
		for (int j = 0; j < mask.col; j++)
			cout << mask.mask[i][j] << ' ';
		cout << endl;
	}
}