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
			resize(targetImg, targetImg, Size(src.rows, src.cols));
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
				offset.at<Vec3f>(i, j)[0] = i;
				offset.at<Vec3f>(i, j)[1] = j;
				offset.at<Vec3f>(i, j)[2] = 0;
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

				offset.at<Vec3f>(i, j)[0] = r_row;
				offset.at<Vec3f>(i, j)[1] = r_col;
				offset.at<Vec3f>(i, j)[2] = MaxDis;
			}
		}

	InitOffsetDis(src, mask, offset);

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
	InitOffsetDis(src, mask, offset);
}

void Inpaint::InitOffsetDis(const Mat& src, const Mat& mask, Mat& offset)
{
	for (int i = 0; src.rows; i++)
		for (int j = 0; src.cols; j++)
		{
			offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, targetImg, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1], mask);

			int iter = 0, maxretry = 20;
			while (offset.at<Vec3f>(i, j)[2] == MaxDis && iter < maxretry)
			{
				iter++;
				offset.at<Vec3f>(i, j)[0] = rand() % src.rows;
				offset.at<Vec3f>(i, j)[1] = rand() % src.cols;
				offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, targetImg, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1], mask);

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

		VoteForTarget(src, mask, offset, true, vote);
		FormTargetImg(src, vote);
		//DeleteVoteArray(vote);
	}
}

void Inpaint::VoteForTarget(const Mat& src, const Mat& mask, const Mat& offset, bool sourceToTarget, int***vote)
{
	targetImg = src.clone();

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

					if (255 == (int)mask.at<uchar>(xs, ys)) continue;

					vote[xt][yt][0] += w * src.at<Vec3b>(xs, ys)[0];
					vote[xt][yt][1] += w * src.at<Vec3b>(xs, ys)[1];
					vote[xt][yt][2] += w * src.at<Vec3b>(xs, ys)[2];
					vote[xt][yt][3] += w;
				}
			}
		}
}

void Inpaint::FormTargetImg(const Mat& src, int ***vote)
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

int*** Inpaint::NewVoteArray(int rows, int cols)
{
	int ***vote = new int**[rows];
	for (int k = 0; k < rows; k++)
		*(vote + k) = new int*[cols];
	for (int k = 0; k < rows; k++)
		for (int l = 0; l < cols; l++)
			*(*(vote + k) + l) = new int[4];

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
			if (xks < 0 || xks >= Src.rows) {dis += ssdmax; continue; }
			if (yks < 0 || yks >= Src.cols) {dis += ssdmax; continue; }

			if (255 == (int)mask.at<uchar>(xks, yks)) {dis += ssdmax; continue; }

			int xkt = xt + dx, ykt = yt + dy;
			if (xkt < 0 || xkt >= Dst.rows) {dis += ssdmax; continue; }
			if (ykt < 0 || ykt >= Dst.cols) {dis += ssdmax; continue; }

			if (255 == (int)mask.at<uchar>(xkt, ykt)) {dis += ssdmax; continue; }
			// SSD distance between pixels (each value is in [0,255^2])
			int ssd = 0;
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

		return (int) (MaxDis * dis / wsum);
}
void Inpaint::Iteration(Mat& src, const Mat& mask, Mat& offset, int iter)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (255 == (int)mask.at<uchar>(i, j))
			{
				Propagation(src, offset, i, j, iter, mask);
				RandomSearch(src, offset, i, j, mask);
			}
		}
}

void Inpaint::Propagation(const Mat& src, Mat& offset, int row, int col, int dir, const Mat& mask)
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
			offset.at < Vec2f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec2f > (row, col)[0], offset.at < Vec2f > (row, col)[1], mask);
			break;
		case 3:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row - 1, col)[0] + 1;
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row - 1, col)[1];
			offset.at < Vec2f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec2f > (row, col)[0], offset.at < Vec2f > (row, col)[1], mask);
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
			offset.at < Vec2f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec2f > (row, col)[0], offset.at < Vec2f > (row, col)[1], mask);
			break;
		case 3:
			offset.at < Vec2f > (row, col)[0] = offset.at < Vec2f > (row + 1, col)[0] - 1;
			offset.at < Vec2f > (row, col)[1] = offset.at < Vec2f > (row + 1, col)[1];
			offset.at < Vec2f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec2f > (row, col)[0], offset.at < Vec2f > (row, col)[1], mask);
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

void Inpaint::RandomSearch(const Mat& src, Mat& offset, int row, int col, const Mat& mask)
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
			offset.at < Vec2f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec2f > (row, col)[0], offset.at < Vec2f > (row, col)[1], mask);
		}
		w /= 2;
	}
}