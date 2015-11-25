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
	void PrintMaskValue(const Mat& mask);
	void PrintOffsetMap(const Mat& offset);
private :
	void BuildPyr();
	void RandomizeOffsetMap(const Mat& src, const Mat& mask, Mat& offset);
	void InitOffsetMap(const Mat& src, const Mat& mask, const Mat& preOff, Mat& offset);
	void InitOffsetDis(const Mat& src, const Mat& mask, Mat& offset);
	void ExpectationMaximization(Mat& src, const Mat& mask, Mat& offset, int level);
	Mat GetPatch(const Mat &Src, int row, int col);
	int Distance(const Mat &Dst, const Mat &Src);
	int Distance(const Mat &Src, int xs, int ys, const Mat &Dst, int xt, int yt, const Mat& mask);
	void Iteration(Mat& src, const Mat& mask, Mat& offset, int iter);
	void Propagation(const Mat& src, Mat& offset, int row, int col, int dir, const Mat& mask);
	void Propagation_Backup(const Mat& src, Mat& offset, int row, int col, int dir, const Mat& mask);
	int GetMinPatch(const Mat& src, const Mat& one, const Mat& two, const Mat& three);
	void RandomSearch(const Mat& src, Mat& offset, int row, int col, const Mat& mask);
	void RandomSearch_Backup(const Mat& src, Mat& offset, int row, int col, const Mat& mask);
	void VoteForTarget(const Mat& src, const Mat& mask, const Mat& offset, bool sourceToTarget, int***vote);
	void FormTargetImg(const Mat& src, int ***vote);
	void BulidSimilarity();
	int*** NewVoteArray(int rows, int cols);

	vector<Mat> srcImg;
	vector<Mat> maskImg;
	vector<Mat> offsetMap;
	Mat targetImg;

	double *similarity;
};
#endif