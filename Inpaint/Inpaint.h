#ifndef INPAINT_H
#define INPAINT_H

#include <iostream>
#include <time.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const int PatchSize = 3;
const int PryLevel = 6;
const int MaxDis = 65535;

class Inpaint
{
public :
	Inpaint(const Mat& src, const Mat& mask);

	void Run();

	// Test
	void PrintMaskValue();

private :
	void BuildPyr();
	void RandomizeOffsetMap(const Mat& src, const Mat& mask, Mat& offset);
	void InitOffsetMap(const Mat& src, const Mat& mask, const Mat& preOff, Mat& offset);
	void ExpectationMaximization(Mat& src, const Mat& mask, Mat& offset, int level);

	vector<Mat> srcImg;
	vector<Mat> maskImg;
	vector<Mat> offsetMap;

};
#endif