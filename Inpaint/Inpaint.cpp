#include "Inpaint.h"

Inpaint::Inpaint(const Mat& src, const Mat& mask)
{
	srand((unsigned)time(NULL));

	assert(src.type() == CV_8UC3);
	assert(mask.type() == CV_8UC1);
	if (!src.data || !mask.data)
	{
		perror("Image is empty!");
		return;
	}

	srcImg.push_back(src.clone());
	maskImg.push_back(mask.clone());
	offsetMap.push_back(Mat(src.size(), CV_32FC2, Scalar::all(0)));
}

void Inpaint::BuildPyr()
{
	// Build gaussian pyramid 01
	buildPyramid(srcImg.front(), srcImg, PryLevel);
	buildPyramid(maskImg.front(), maskImg, PryLevel);
	buildPyramid(offsetMap.front(), offsetMap, PryLevel);

	/*
	// Show the pyr
	vector<Mat>::iterator itbg = srcImg.begin();
	vector<Mat>::iterator itend = srcImg.end();

	int i = 0;
	std::stringstream title;
	for(; itbg < itend; ++itbg){
		title << "Gaussian Pyramid " << i;
		namedWindow(title.str());
		imshow(title.str(), *itbg);
		++i;
		title.clear();
	}
	*/
}

void Inpaint::Run()
{
	vector<Mat>::iterator srcItBg = srcImg.begin();
	vector<Mat>::iterator srcItEnd = srcImg.end();
	vector<Mat>::iterator maskItBg = maskImg.begin();
	vector<Mat>::iterator maskItEnd = maskImg.end();

	for (; srcItEnd > srcItBg; srcItEnd--)
	{
		Mat src = *srcItEnd;
		Mat mask = *maskItEnd;

		maskItEnd--;
	}
}