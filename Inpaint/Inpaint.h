#ifndef INPAINT_H
#define INPAINT_H

#include <iostream>
#include <time.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#define MY_DEBUG
//#define SHOW_INTERMEDIATE
using namespace cv;
using namespace std;

const int PatchSize = 3;
const int PryLevel = 8;
const int MaxDis = 65535;

struct NNF
{
	Mat input;
	Mat output;
	Mat offset;
};

struct Mask
{
	bool** mask;
	int row;
	int col;
	Mask(bool** mask, int row, int col)
	{
		this->mask = mask;
		this->row = row;
		this->col = col;
	}
	
};

struct MaskedImage
{
	Mat mask;
	Mat img;
	int row;
	int col;
	MaskedImage(const Mat& img, const Mat& mask)
	{
		this->mask = mask.clone();
		this->img = img.clone();
		row = img.rows;
		col = img.cols;
	}
	MaskedImage()
	{
		mask = NULL;
		img = NULL;
		row = 0;
		col = 0;
	}
	MaskedImage(int w, int h)
	{
		this->row = w;
		this->col = h;
		this->mask = Mat(Size(h, w), CV_8UC1, Scalar::all(0));
		this->img = Mat(Size(h, w), CV_8UC3, Scalar::all(0));
	}
	MaskedImage copy()
	{
		return MaskedImage(this->img, this->mask);
	}
	void SetMask(int i, int j, int value)
	{
		mask.at<uchar>(i, j) = value;
	}
	bool IsMasked(int i, int j) const
	{
		if (1 == (int)mask.at<uchar>(i, j))
			return true;
		else
			return false;
	}
	bool ContainsMask(int i, int j)
	{
		for (int dx = -(PatchSize / 2); dx <= (PatchSize / 2); dx++)
			for (int dy = -(PatchSize / 2); dy <= (PatchSize / 2); dy++)
			{
				int xs = i + dx, ys = j + dy;
				if (xs < 0 || xs >= row || ys < 0 || ys >= col)
					continue;
				if (1 == (int)mask.at<uchar>(xs, ys))
					return true;
			}
		return false;
	}
	MaskedImage Upscale(int newW, int newH)
	{
		MaskedImage newimage(newW, newH);

		for (int i = 0; i < newW; i++)
			for (int j = 0; j < newH; j++)
			{
				int xs = (i * row) / newW;
				int ys = (j * col) / newH;

				if (0 == (int)mask.at<uchar>(xs, ys))
				{
					newimage.img.at<Vec3b>(i, j)[0] = this->img.at<Vec3b>(xs, ys)[0];
					newimage.img.at<Vec3b>(i, j)[1] = this->img.at<Vec3b>(xs, ys)[1];
					newimage.img.at<Vec3b>(i, j)[2] = this->img.at<Vec3b>(xs, ys)[2];
					newimage.SetMask(i, j, 0);
				}
				else
					newimage.SetMask(i, j, 1);
			}
		return newimage;
	}
};
class Inpaint
{
public :
	Inpaint(const Mat& src, const Mat& mask);

	void Run();

	// Test
	void PrintMaskValue();
	void PrintMaskValue(const Mat& mask);
	void PrintMaskValue(const Mask& mask);
	void PrintOffsetMap(const Mat& offset);

private :
	void BuildPyr();
	void RandomizeOffsetMap(const MaskedImage& src, const MaskedImage& target, Mat& offset);
	void InitOffsetMap(const MaskedImage& src, const MaskedImage& target, const Mat& preOff, Mat& offset);
	void InitOffsetDis(const MaskedImage& src, const MaskedImage& target, Mat& offset);
	void ExpectationMaximization(MaskedImage& src, MaskedImage& target, Mat& offset_SourceToTarget, Mat& offset_TargetToSource, int level);
	Mat GetPatch(const Mat &Src, int row, int col);
	int Distance(const Mat &Dst, const Mat &Src);
	int Distance(const MaskedImage &Src, int xs, int ys, const MaskedImage &Dst, int xt, int yt);
	void Iteration(MaskedImage& src, MaskedImage& target, Mat& offset, int iter);
	void Propagation(const MaskedImage& src, const MaskedImage& tar, Mat& offset, int row, int col, int dir);
	void Propagation_Backup(const Mat& src, Mat& offset, int row, int col, int dir, const Mat& mask);
	int GetMinPatch(const Mat& src, const Mat& one, const Mat& two, const Mat& three);
	void RandomSearch(const MaskedImage& src, const MaskedImage& tar, Mat& offset, int row, int col);
	void RandomSearch_Backup(const Mat& src, Mat& offset, int row, int col, const Mat& mask);
	void VoteForTarget(const MaskedImage& src, const MaskedImage& tar, const Mat& offset, bool sourceToTarget, Mat vote, bool upscale, const const MaskedImage& newsrc);
	void WeightedCopy(const MaskedImage& src, int xs, int ys, Mat vote, int xd, int yd, double w);
	void FormTargetImg(MaskedImage& target, Mat vote);
	void BulidSimilarity();
	double*** NewVoteArray(int rows, int cols);
	Mask BulidMask(const Mat& mask);
	bool** NewMask(int row, int col);
	Mask DownsampleMask(Mask mask, int row, int col);

	vector<Mat> srcImg;
	vector<MaskedImage> maskedImage;
	Mat originMask;

	vector<Mat> maskImg;
	vector<Mat> maskImg_TargetToSource;
	vector<Mask> mask_st;
	vector<Mask> mask_ts;

	vector<Mat> offsetMap_TargetToSource;
	vector<Mat> offsetMap_SourceToTarget;
	//Mat targetImg;
	MaskedImage target;
	NNF sourceToTarget;
	NNF targetToSource;
	double *similarity;
};
#endif