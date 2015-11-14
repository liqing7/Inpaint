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

class Inpaint
{
public :
	Inpaint(const Mat& src, const Mat& mask);

	void Run();
	void BuildPyr();
	
private :
	vector<Mat> srcImg;
	vector<Mat> maskImg;
	vector<Mat> offsetMap;

};
#endif