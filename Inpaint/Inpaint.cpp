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
	maskImg_SourceToTarget.push_back(mask.clone());
	offsetMap_SourceToTarget.push_back(Mat(src.size(), CV_32FC3, Scalar::all(0)));
	offsetMap_TargetToSource.push_back(Mat(src.size(), CV_32FC3, Scalar::all(0)));
	BulidSimilarity();
	//test
	BuildPyr();
}

void BulidMask(const Mat& mask)
{
	bool** mask = new bool*[mask.rows];
	for (int i = 1; i < mask.rows; i++)



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
	buildPyramid(offsetMap_SourceToTarget.front(), offsetMap_SourceToTarget, PryLevel);
	buildPyramid(offsetMap_TargetToSource.front(), offsetMap_TargetToSource, PryLevel);

	/*
	// Show the pyr
	vector<Mat>::iterator itbg = srcImg.begin();
	vector<Mat>::iterator itend = srcImg.end();
	
	int i = 0;
	std::stringstream title;
	Mat tmp;
	for(; itbg < itend; ++itbg){
		title << "Gaussian Pyramid " << i;
		namedWindow(title.str());
		resize(*itbg, tmp, Size(srcImg.front().cols, srcImg.front().rows));
		imshow(title.str(), tmp);
		++i;
		title.clear();
	}
	waitKey();
	*/
}

void Inpaint::Run()
{
	BuildPyr();

	vector<Mat>::iterator srcItBg = srcImg.begin();
	vector<Mat>::iterator srcItEnd = srcImg.end()-1;
	vector<Mat>::iterator maskItBg = maskImg.begin();
	vector<Mat>::iterator maskItEnd = maskImg.end()-1;
	vector<Mat>::iterator offsetMapBg_SourceToTarget = offsetMap_SourceToTarget.begin();
	vector<Mat>::iterator offsetMapEnd_SourceToTarget = offsetMap_SourceToTarget.end()-1;
	vector<Mat>::iterator offsetMapBg_TargetToSource = offsetMap_TargetToSource.begin();
	vector<Mat>::iterator offsetMapEnd_TargetToSource = offsetMap_TargetToSource.end()-1;

	Mat tempImg;
	int index = srcImg.size();
	for (; srcItEnd >= srcItBg; srcItEnd--, maskItEnd--, offsetMapEnd_SourceToTarget--, offsetMapEnd_TargetToSource--, index--)
	{
		Mat src = *srcItEnd;
		Mat mask = *maskItEnd;
		Mat offset_TargetToSource = *offsetMapEnd_TargetToSource;
		Mat offset_SourceToTarget = *offsetMapEnd_SourceToTarget;

		cout << "Pry " << index << endl;
		if (srcItEnd == srcImg.end() - 1)
		{
			// Initialize offsetmap with random values
			targetImg = src.clone();
			RandomizeOffsetMap(src, targetImg, mask, offset_SourceToTarget);
			RandomizeOffsetMap(targetImg, src, mask, offset_TargetToSource);
		}
		else
		{	
			resize(targetImg, tempImg, Size((*srcItBg).cols, (*srcItBg).rows));
			resize(targetImg, targetImg, Size(src.cols, src.rows));
			imshow("Pry " + index, tempImg);
			waitKey();
			//return;
			// Initialize offsetmap with the small offsetmap
			InitOffsetMap(src, targetImg, mask, *(offsetMapEnd_SourceToTarget+1), offset_SourceToTarget);
			InitOffsetMap(targetImg, src, mask, *(offsetMapEnd_TargetToSource+1), offset_TargetToSource);
		}

		//EM-like
		ExpectationMaximization(src, targetImg, mask, offset_SourceToTarget, offset_TargetToSource, index);
		if (srcItEnd == srcItBg)
		{
			cout << "ok in here" << endl;
			break;
		}
	}
	imshow("Pry " + index, targetImg);
	cout << "END!!!" << endl;
	waitKey();
}

void Inpaint::RandomizeOffsetMap(const Mat& src, const Mat& target, const Mat& mask, Mat& offset)
{
#ifdef MY_DEBUG
	PrintMaskValue(mask);
#endif
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (150 >= (int)mask.at<uchar>(i, j))
			{
				// Need not search
				offset.at<Vec3f>(i, j)[0] = i;
				offset.at<Vec3f>(i, j)[1] = j;
				offset.at<Vec3f>(i, j)[2] = 0;
			}
			else 
			{
				int r_col = rand() % src.cols;
				int r_row = rand() % src.rows;

				while (150 < (int)mask.at<uchar>(r_row, r_col))
				{
					r_col = rand() % src.cols;
					r_row = rand() % src.rows;
				}

				offset.at<Vec3f>(i, j)[0] = r_row;
				offset.at<Vec3f>(i, j)[1] = r_col;
				offset.at<Vec3f>(i, j)[2] = MaxDis;
			}
		}

	InitOffsetDis(src, target, mask, offset);
}

void Inpaint::InitOffsetMap(const Mat& src, const Mat& target, const Mat& mask, const Mat& preOff, Mat& offset)
{
	int fx = offset.rows / preOff.rows;
	int fy = offset.cols / preOff.cols;

	fx = fx == 1 ? 2 : fx;
	fy = fy == 1 ? 2 : fy;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (150 >= (int)mask.at<uchar>(i, j))
			{
				// Need not search
				offset.at<Vec3f>(i, j)[0] = i;
				offset.at<Vec3f>(i, j)[1] = j;
				offset.at<Vec3f>(i, j)[2] = 0;
			}
			else 
			{
				int xlow = i / fx;
				int ylow = j / fy;
				offset.at<Vec3f>(i, j)[0] = preOff.at<Vec3f>(xlow, ylow)[0] * fx;
				offset.at<Vec3f>(i, j)[1] = preOff.at<Vec3f>(xlow, ylow)[1] * fy;
				offset.at<Vec3f>(i, j)[2] = MaxDis;
			}
		}
	InitOffsetDis(src, target, mask, offset);
}

void Inpaint::InitOffsetDis(const Mat& src, const Mat& target, const Mat& mask, Mat& offset)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (150 >= (int)mask.at<uchar>(i, j))
			{
				// Need not search
				offset.at<Vec3f>(i, j)[0] = i;
				offset.at<Vec3f>(i, j)[1] = j;
				offset.at<Vec3f>(i, j)[2] = 0;
				continue;
			}

			offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, target, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1], mask);

			int iter = 0, maxretry = 20;
			while (offset.at<Vec3f>(i, j)[2] > 8000 && iter < maxretry)
			{
				iter++;
				offset.at<Vec3f>(i, j)[0] = rand() % src.rows;
				offset.at<Vec3f>(i, j)[1] = rand() % src.cols;
				offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, target, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1], mask);

			}
		}
	//PrintOffsetMap(offset);
}

void Inpaint::ExpectationMaximization(Mat& src, Mat& target, const Mat& mask, Mat& offset_SourceToTarget, Mat& offset_TargetToSource, int level)
{
	int iterEM = 1 + 2 * level;
	int iterNNF = 1 + level;

	for (int i = 0; i < iterEM; i++)
	{
		cout << "ITER " << i << endl;
#ifdef MY_DEBUG
		PrintOffsetMap(offset_TargetToSource);
#endif
		// PatchMatch
		for (int j = 0; j < iterNNF; j++)
		{
			Iteration(src, target, mask, offset_SourceToTarget, j);
			Iteration(target, src, mask, offset_TargetToSource, j);
		}
#ifdef MY_DEBUG
		PrintOffsetMap(offset_TargetToSource);
#endif
		// Form a target image
		// New a vote array
		double ***vote = NewVoteArray(src.rows, src.cols);
		bool upscale = false;
		if (i == iterEM - 1)
			upscale = true;
		VoteForTarget(src, target, mask, offset_SourceToTarget, true, vote, upscale);
		VoteForTarget(target, src, mask, offset_TargetToSource, false, vote, upscale);
		FormTargetImg(src, vote);
		//DeleteVoteArray(vote);
	}
}

void Inpaint::VoteForTarget(const Mat& src, const Mat& tar, const Mat& mask, const Mat& offset, bool sourceToTarget, double***vote, bool upscale)
{
	//targetImg = src.clone();

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			int xp = offset.at<Vec3f>(i, j)[0], yp = offset.at<Vec3f>(i, j)[1], dp = offset.at<Vec3f>(i, j)[2];

			double w = similarity[dp];

			for (int dx = -(PatchSize / 2); dx <= PatchSize / 2; dx++)
			{
				for (int dy = -(PatchSize / 2); dy <= PatchSize / 2; dy++)
				{
					int xs, ys, xt, yt;
					if (sourceToTarget) 
						{ xs = i + dx; ys = j + dy;	xt = xp + dx; yt = yp + dy;	}
					else
						{ xs = xp + dx; ys = yp + dy; xt = i + dx; yt = j + dy; }

					if (xs < 0 || xs >= src.rows) continue;
					if (ys < 0 || ys >= src.cols) continue;
					if (xt < 0 || xt >= src.rows) continue;
					if (yt < 0 || yt >= src.cols) continue;

					if (150 < (int)mask.at<uchar>(xs, ys)) continue;

				//	if (upscale)
				//	{
				//		WeightedCopy(src, 2*xs, 2*ys, vote, 2*xt, 2*yt, w);
				//		WeightedCopy(src, 2*xs+1, 2*ys, vote, 2*xt+1, 2*yt, w);
				//		WeightedCopy(src, 2*xs, 2*ys+1, vote, 2*xt, 2*yt+1, w);
				//		WeightedCopy(src, 2*xs+1, 2*ys+1, vote, 2*xt+1, 2*yt+1, w);
				//	}
					//else 
						WeightedCopy(src, xs, ys, vote, xt, yt, w); 
				}
			}
		}
}

void Inpaint::WeightedCopy(const Mat& src, int xs, int ys, double ***vote, int xd, int yd, double w)
{
	vote[xd][yd][0] += w * src.at<Vec3b>(xs, ys)[0];
	vote[xd][yd][1] += w * src.at<Vec3b>(xs, ys)[1];
	vote[xd][yd][2] += w * src.at<Vec3b>(xs, ys)[2];
	vote[xd][yd][3] += w;
}

void Inpaint::FormTargetImg(const Mat& src, double ***vote)
{
	for (int i = 0; i < targetImg.rows; i++)
	{
		for (int j = 0; j < targetImg.cols; j++)
		{
			if (vote[i][j][3] > 0)
			{
				targetImg.at<Vec3b>(i, j)[0] = (int)(vote[i][j][0] / vote[i][j][3]);
				targetImg.at<Vec3b>(i, j)[1] = (int)(vote[i][j][1] / vote[i][j][3]);
				targetImg.at<Vec3b>(i, j)[2] = (int)(vote[i][j][2] / vote[i][j][3]);
			}
		}
	}
}

double*** Inpaint::NewVoteArray(int rows, int cols)
{
	double ***vote = new double**[rows];
	for (int k = 0; k < rows; k++)
		*(vote + k) = new double*[cols];
	for (int k = 0; k < rows; k++)
		for (int l = 0; l < cols; l++)
		{
			*(*(vote + k) + l) = new double[4];
			for (int m = 0; m < 4; m++)
				vote[k][l][m] = 0;
		}

	return vote;
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

int Inpaint::Distance(const Mat &Src, int xs, int ys, const Mat &Dst, int xt, int yt, const Mat& mask)
{
	int dis = 0, wsum = 0, ssdmax = 255 * 255 * 9;

	for (int dy = -(PatchSize / 2); dy <= PatchSize / 2; dy++)
		for (int dx = -(PatchSize / 2); dx <= PatchSize / 2; dx++)
		{
			wsum += ssdmax;

			int xks = xs + dx, yks = ys + dy;
			if (xks < 1 || xks >= Src.rows - 1) {dis += ssdmax; continue; }
			if (yks < 1 || yks >= Src.cols - 1) {dis += ssdmax; continue; }

			if (150 < (int)mask.at<uchar>(xks, yks)) {dis += ssdmax; continue; }

			int xkt = xt + dx, ykt = yt + dy;
			if (xkt < 1 || xkt >= Dst.rows - 1) {dis += ssdmax; continue; }
			if (ykt < 1 || ykt >= Dst.cols - 1) {dis += ssdmax; continue; }

			if (150 < (int)mask.at<uchar>(xkt, ykt)) {dis += ssdmax; continue; }
			// SSD distance between pixels (each value is in [0,255^2])
			long long ssd = 0;
			for(int band = 0; band < 3; band++) {
				// pixel values
				int s_value = Src.at<Vec3b>(xks, yks)[band];
				int t_value = Src.at<Vec3b>(xkt, ykt)[band];
					
				// pixel horizontal gradients (Gx)
				int s_gx = 128+(Src.at<Vec3b>(xks+1, yks)[band] - Src.at<Vec3b>(xks-1, yks)[band])/2;
				int t_gx = 128+(Dst.at<Vec3b>(xkt+1, ykt)[band] - Dst.at<Vec3b>(xkt-1, ykt)[band])/2;

				// pixel vertical gradients (Gy)
				int s_gy = 128+(Src.at<Vec3b>(xks, yks+1)[band] - Src.at<Vec3b>(xks, yks-1)[band])/2;
				int t_gy = 128+(Dst.at<Vec3b>(xkt, ykt+1)[band] - Dst.at<Vec3b>(xkt, ykt-1)[band])/2;

				ssd += (s_value-t_value) * (s_value-t_value); // distance between values in [0,255^2]
				ssd += (s_gx-t_gx) * (s_gx-t_gx); // distance between Gx in [0,255^2]
				ssd += (s_gy-t_gy) * (s_gy-t_gy); // distance between Gy in [0,255^2]
			}

			// add pixel distance to global patch distance
			dis += ssd;
		}

		return (int) ((long long) MaxDis * dis / (long long)wsum);
}
void Inpaint::Iteration(Mat& src, Mat& target, const Mat& mask, Mat& offset, int iter)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (150 < (int)mask.at<uchar>(i, j))
			{
				Propagation(src, target, offset, i, j, iter, mask);
				RandomSearch(src, target, offset, i, j, mask);
			}
		}
}

void Inpaint::Propagation(const Mat& src, const Mat& tar, Mat& offset, int row, int col, int dir, const Mat& mask)
{
	int xp, yp, dp;
	dir %= 2;
	if (0 == dir) dir = -1;

	if (col - dir > 0 && col - dir < src.cols)
	{
		xp = offset.at<Vec3f>(row, col - dir)[0];
		yp = offset.at<Vec3f>(row, col - dir)[1] + dir;
		dp = Distance(src, row, col, tar, xp, yp, mask);

		if (dp < offset.at<Vec3f>(row, col)[2])
		{
			offset.at<Vec3f>(row, col)[0] = xp;
			offset.at<Vec3f>(row, col)[1] = yp;
			offset.at<Vec3f>(row, col)[2] = dp;
		}
	}

	if (row - dir > 0 && row - dir < src.rows)
	{
		xp = offset.at<Vec3f>(row - dir, col)[0] + dir;
		yp = offset.at<Vec3f>(row - dir, col)[1];
		dp = Distance(src, row, col, tar, xp, yp, mask);

		if (dp < offset.at<Vec3f>(row, col)[2])
		{
			offset.at<Vec3f>(row, col)[0] = xp;
			offset.at<Vec3f>(row, col)[1] = yp;
			offset.at<Vec3f>(row, col)[2] = dp;
		}
	}

}
void Inpaint::Propagation_Backup(const Mat& src, Mat& offset, int row, int col, int dir, const Mat& mask)
{
	Mat DstPatch = GetPatch(targetImg, row, col);
	Mat SrcPatch = GetPatch(src, offset.at<Vec3f>(row, col)[0], offset.at<Vec3f>(row, col)[1]);
	Mat LeftPatch, RightPatch, UpPatch, DownPatch;

	if (0 == dir % 2)
	{
		if (col - 1 >= 0)
			LeftPatch = GetPatch(src, offset.at<Vec3f>(row, col - 1)[0], offset.at<Vec3f>(row, col - 1)[1] + 1);
		if (row - 1 >= 0)
			UpPatch = GetPatch(src, offset.at<Vec3f>(row - 1, col)[0] + 1, offset.at<Vec3f>(row - 1, col)[1] + 1);

		int location = GetMinPatch(DstPatch, SrcPatch, LeftPatch, UpPatch);

		switch (location)
		{
		case 2:
			offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row, col - 1)[0];
			offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row, col - 1)[1] + 1;
			offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
			break;
		case 3:
			offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row - 1, col)[0] + 1;
			offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row - 1, col)[1];
			offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
			break;
		}
	}
	else 
	{
		if (col + 1 < src.cols)
			RightPatch = GetPatch(src, offset.at<Vec3f>(row, col + 1)[0], offset.at<Vec3f>(row, col + 1)[1] - 1);
		if (row + 1 < src.rows)
			DownPatch = GetPatch(src, offset.at<Vec3f>(row + 1, col)[0] - 1, offset.at<Vec3f>(row + 1, col)[1] - 1);

		int location = GetMinPatch(DstPatch, SrcPatch, RightPatch, DownPatch);

		switch (location)
		{
		case 2:
			offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row, col + 1)[0];
			offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row, col + 1)[1] - 1;
			offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
			break;
		case 3:
			offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row + 1, col)[0] - 1;
			offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row + 1, col)[1];
			offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
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

void Inpaint::RandomSearch_Backup(const Mat& src, Mat& offset, int row, int col, const Mat& mask)
{
	Mat DstPatch = GetPatch(targetImg, row, col);
	Mat SrcPatch = GetPatch(src, offset.at<Vec3f>(row, col)[0], offset.at<Vec3f>(row, col)[1]);

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
			offset.at < Vec3f > (row, col)[0] = x;
			offset.at < Vec3f > (row, col)[1] = y;
			offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
		}
		w /= 2;
	}
}

void Inpaint::RandomSearch(const Mat& src, const Mat& tar, Mat& offset, int row, int col, const Mat& mask)
{
	int w = min(src.cols, src.rows);
	
	while (w > 0)
	{
		int x = rand() % w + row;
		int y = rand() % w + col;
		x = max(0, min(src.rows, x));
		y = max(0, min(src.cols, y));

		int d = Distance(src, row, col, tar, x, y, mask);

		if (d < offset.at < Vec3f > (row, col)[2])
		{
			offset.at < Vec3f > (row, col)[0] = x;
			offset.at < Vec3f > (row, col)[1] = y;
			offset.at < Vec3f > (row, col)[2] = d;
		}
		w /= 2;
	}
}