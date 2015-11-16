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
	vector<Mat>::iterator srcItEnd = srcImg.end();
	vector<Mat>::iterator maskItBg = maskImg.begin();
	vector<Mat>::iterator maskItEnd = maskImg.end();
	vector<Mat>::iterator offsetMapBg = offsetMap.begin();
	vector<Mat>::iterator offsetMapEnd = offsetMap.end();

	int index = srcImg.size();
	for (; srcItEnd > srcItBg; srcItEnd--, maskItEnd--, offsetMapEnd--, index--)
	{
		Mat src = *srcItEnd;
		Mat mask = *maskItEnd;
		Mat offset = *offsetMapEnd;

		cout << "Pry " << index << endl;
		if (srcItEnd == srcImg.end())
			// Initialize offsetmap with random values
			RandomizeOffsetMap(src, mask, offset);
		else
			// Initialize offsetmap with the small offsetmap
			InitOffsetMap(src, mask, *(offsetMapEnd+1), offset);

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
				offset.at<Vec2f>(i, j)[0] = 0;
				offset.at<Vec2f>(i, j)[1] = 0;
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
		
	}
}