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
	offsetMap.push_back(Mat(src.size(), CV_32FC3, Scalar::all(0)));
	BulidSimilarity();
}

void Inpaint::BulidSimilarity()
{
	similarity = new double[MaxDis + 1];
	double base[] = {1.0, 0.99, 0.96, 0.83, 0.38, 0.11, 0.02, 0.005, 0.0006, 0.0001, 0 };

	// stretch base array 
	for(int i = 0; i < MaxDis + 1; i++) 
	{
		double t = (double)i / (MaxDis + 1);

		// interpolate from base array values
		int j = (int)(100 * t), k = j + 1;
		double vj = (j < 10) ? base[j] : 0;
		double vk = (k < 10) ? base[k] : 0;
			
		double v = vj + (100*t-j)*(vk-vj);
		similarity[i] = v;
	}
}

void Inpaint::BuildPyr()
{
	// Build gaussian pyramid 
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
	BuildPyr();

	vector<Mat>::iterator srcItBg = srcImg.begin();
	vector<Mat>::iterator srcItEnd = srcImg.end()-1;
	vector<Mat>::iterator maskItBg = maskImg.begin();
	vector<Mat>::iterator maskItEnd = maskImg.end()-1;
	vector<Mat>::iterator offsetMapBg = offsetMap.begin();
	vector<Mat>::iterator offsetMapEnd = offsetMap.end()-1;

	int index = srcImg.size();
	for (; srcItEnd > srcItBg; srcItEnd--, maskItEnd--, offsetMapEnd--, index--)
	{
		Mat src = *srcItEnd;
		Mat mask = *maskItEnd;
		Mat offset = *offsetMapEnd;

		cout << "Pry " << index << endl;
		if (srcItEnd == srcImg.end() - 1)
		{
			// Initialize offsetmap with random values
			targetImg = src.clone();
			RandomizeOffsetMap(src, mask, offset);
		}
		else
		{
			// Initialize offsetmap with the small offsetmap
			InitOffsetMap(src, mask, *(offsetMapEnd+1), offset);
		}

		//EM-like
		ExpectationMaximization(src, mask, offset, index);
	}
}

void Inpaint::RandomizeOffsetMap(const Mat& src, const Mat& mask, Mat& offset)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (0 == (int)mask.at<uchar>(i, j))
			{
				// Need not search
				offset.at<Vec2f>(i, j)[0] = i;
				offset.at<Vec2f>(i, j)[1] = j;
			}
			else 
			{
				int r_col = rand() % src.cols;
				int r_row = rand() % src.rows;

				while (255 == (int)mask.at<uchar>(i, j))
				{
					r_col = rand() % src.cols;
					r_row = rand() % src.rows;
				}

				offset.at<Vec2f>(i, j)[0] = r_row;
				offset.at<Vec2f>(i, j)[1] = r_col;
			}
		}

}

void Inpaint::InitOffsetMap(const Mat& src, const Mat& mask, const Mat& preOff, Mat& offset)
{
	int fx = offset.rows / preOff.rows;
	int fy = offset.cols / preOff.cols;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (0 == (int)mask.at<uchar>(i, j))
			{
				// Need not search
				offset.at<Vec2f>(i, j)[0] = 0;
				offset.at<Vec2f>(i, j)[1] = 0;
			}
			else 
			{
				int xlow = i / fx;
				int ylow = j / fy;
				offset.at<Vec2f>(i, j)[0] = preOff.at<Vec2f>(xlow, ylow)[0] * fx;
				offset.at<Vec2f>(i, j)[1] = preOff.at<Vec2f>(xlow, ylow)[1] * fy;
			}
		}
}

void Inpaint::ExpectationMaximization(Mat& src, const Mat& mask, Mat& offset, int level)
{
	int iterEM = 1 + 2 * level;
	int iterNNF = 1 + level;

	for (int i = 0; i < iterEM; i++)
	{
		// PatchMatch
		for (int j = 0; j < iterNNF; j++)
			Iteration(src, mask, offset, j);

		// Form a target image
		// New a vote array
		int ***vote = NewVoteArray(src.rows, src.cols);

		VoteForTarget(src, mask, offset);
		//DeleteVoteArray(vote);
	}
}

void Inpaint::VoteForTarget(const Mat& src, const Mat& mask, const Mat& offset)
{
	targetImg = src.clone();

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{

		}
}

int*** NewVoteArray(int rows, int cols)
{
	int ***vote = new int**[rows];
	for (int k = 0; k < rows; k++)
		*(vote + k) = new int*[cols];
	for (int k = 0; k < rows; k++)
		for (int l = 0; l < cols; l++)
			*(*(vote + k) + l) = new int[4];
}

void DeleteVoteArray(int*** vote)
{

}

Mat Inpaint::GetPatch(const Mat &Src, int row, int col)
{
	int row_begin = row - (PatchSize / 2) >= 0 ? row - (PatchSize / 2) : 0;
	int row_end =
		row + (PatchSize / 2) <= Src.rows - 1 ?
		row + (PatchSize / 2) : Src.rows - 1;

	int col_begin = col - (PatchSize / 2) >= 0 ? col - (PatchSize / 2) : 0;
	int col_end =
		col + (PatchSize / 2) <= Src.cols - 1 ?
		col + (PatchSize / 2) : Src.cols - 1;

	return Src(Range(row_begin, row_end + 1), Range(col_begin, col_end + 1));
}

int Inpaint::Distance(const Mat &Dst, const Mat &Src)
{
	int distance = 0;

	for (int i = 0; i < Dst.rows; i++)
	{
		for (int j = 0; j < Dst.cols; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				int tem = Src.at < Vec3b > (i, j)[k] - Dst.at < Vec3b > (i, j)[k];
				distance += tem * tem;
			}
		}
	}

	return distance;
}

int Inpaint::Distance(const Mat &Src, int xs, int ys, const Mat &Dst, int xt, int yt)
{
	int dis = 0, wsum = 0, ssdmax = 255 * 255 * PatchSize * PatchSize;

	for (int dy = -(PatchSize / 2); dy <= PatchSize / 2; dy++)
		for (int dx = -(PatchSize / 2); dx <= PatchSize / 2; dx++)
		{

		}


}
void Inpaint::Iteration(Mat& src, const Mat& mask, Mat& offset, int iter)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (255 == (int)mask.at<uchar>(i, j))
			{
				Propagation(src, offset, i, j, iter);
				RandomSearch(src, offset, i, j);
			}
		}
}

void Inpaint::Propagation(const Mat& src, Mat& offset, int row, int col, int dir)
{
	Mat DstPatch = GetPatch(targetImg, row, col);
	Mat SrcPatch = GetPatch(src, offset.at<Vec2f>(row, col)[0], offset.at<Vec2f>(row, col)[1]);
	Mat LeftPatch, RightPatch, UpPatch, DownPatch;

	if (0 == dir % 2)
	{
		if (col - 1 >= 0)
			LeftPatch = GetPatch(src, offset.at<Vec2f>(row, col - 1)[0], offset.at<Vec2f>(row, col - 1)[1] + 1);
		if (row - 1 >= 0)
			UpPatch = GetPatch(src, offset.at<Vec2f>(row - 1, col)[0] + 1, offset.at<Vec2f>(row - 1, col)[1] + 1);

		int location = GetMinPatch(DstPatch, SrcPatch, LeftPatch, UpPatch);

		switch (location)
		{
		case 2:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row, col - 1)[0];
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row, col - 1)[1] + 1;
			break;
		case 3:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row - 1, col)[0] + 1;
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row - 1, col)[1];
			break;
		}
	}
	else 
	{
		if (col + 1 < src.cols)
			RightPatch = GetPatch(src, offset.at<Vec2f>(row, col + 1)[0], offset.at<Vec2f>(row, col + 1)[1] - 1);
		if (row + 1 < src.rows)
			DownPatch = GetPatch(src, offset.at<Vec2f>(row + 1, col)[0] - 1, offset.at<Vec2f>(row + 1, col)[1] - 1);

		int location = GetMinPatch(DstPatch, SrcPatch, RightPatch, DownPatch);

		switch (location)
		{
		case 2:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row, col + 1)[0];
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row, col + 1)[1] - 1;
			break;
		case 3:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row + 1, col)[0] - 1;
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row + 1, col)[1];
			break;
		}
	}
}

int Inpaint::GetMinPatch(const Mat& src, const Mat& one, const Mat& two, const Mat& three)
{
	int dis1 = Distance(src, one);
	int dis2 = Distance(src, two);
	int dis3 = Distance(src, three);

	if (dis1 <= dis2 && dis1 <= dis3)
		return 1;

	if (dis2 <= dis1 && dis2 <= dis3)
		return 2;

	return 3;
}

void Inpaint::RandomSearch(const Mat& src, Mat& offset, int row, int col)
{
	Mat DstPatch = GetPatch(targetImg, row, col);
	Mat SrcPatch = GetPatch(src, offset.at<Vec2f>(row, col)[0], offset.at<Vec2f>(row, col)[1]);

	int w = min(src.cols, src.rows);
	
	while (w > 0)
	{
		int x = rand() % w;
		int y = rand() % w;

		Mat candidate = GetPatch(src, x, y);

		int dis1 = Distance(SrcPatch, DstPatch);
		int dis2 = Distance(SrcPatch, candidate);

		if (dis2 < dis1)
		{
			offset.at < Vec2f > (row, col)[0] = x;
			offset.at < Vec2f > (row, col)[1] = y;
		}
		w /= 2;
	}
}