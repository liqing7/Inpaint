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
	originMask = mask.clone();
	//maskImg_SourceToTarget.push_back(mask.clone());
	//mask_st.push_back(BulidMask(mask));
	offsetMap_SourceToTarget.push_back(Mat(src.size(), CV_32FC3, Scalar::all(0)));
	offsetMap_TargetToSource.push_back(Mat(src.size(), CV_32FC3, Scalar::all(0)));
	BulidSimilarity();
	//test
	//BuildPyr();
}

Mask Inpaint::BulidMask(const Mat& mask)
{
	bool** tempmask = new bool*[mask.rows];
	for (int i = 0; i < mask.rows; i++)
		tempmask[i] = new bool[mask.cols];
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
			if (255 == (int)mask.at<uchar>(i, j))
				tempmask[i][j] = true;
			else
				tempmask[i][j] = false;
	
	return Mask(tempmask, mask.rows, mask.cols);
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
bool** Inpaint::NewMask(int row, int col)
{
	bool** tempmask = new bool*[row];
	for (int i = 0; i < row; i++)
		tempmask[i] = new bool[col];

	return tempmask;
}

Mask Inpaint::DownsampleMask(Mask mask, int row, int col)
{
	bool** tempmask = NewMask(row, col);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		{
			int x1 = i * 2, x2 = i * 2 + 1, y1 = j * 2, y2 = j * 2 + 1;
			if (x1 >= mask.row || y1 >= mask.col || x2 >= mask.row || y2 >= mask.col)
			{
				tempmask[i][j] = false;
				continue;
			}

			if (mask.mask[x1][y1] || mask.mask[x1][y2] || mask.mask[x2][y1] || mask.mask[x2][y2])
				tempmask[i][j] = true;
			else
				tempmask[i][j] = false;
		}
	return Mask(tempmask, row, col);
}

void Inpaint::BuildPyr()
{
	int x = srcImg.front().rows, y = srcImg.front().cols;
	int level = 0;
	while (x > PatchSize && y > PatchSize)
	{
		x >>= 1;
		y >>= 1;
		level++;
	}
	// Build gaussian pyramid 
	for (int i = 0; i < originMask.rows; i++)
		for (int j = 0; j < originMask.cols; j++)
			if (255 == (int)originMask.at<uchar>(i, j))
				originMask.at<uchar>(i, j) = 1;
	maskImg.push_back(originMask);

	buildPyramid(srcImg.front(), srcImg, level);
	buildPyramid(maskImg.front(), maskImg, PryLevel);
	buildPyramid(offsetMap_SourceToTarget.front(), offsetMap_SourceToTarget, level);
	buildPyramid(offsetMap_TargetToSource.front(), offsetMap_TargetToSource, level);

	// Bulid MaskedImage pyr
	vector<Mat>::iterator itbg = srcImg.begin();
	vector<Mat>::iterator itend = srcImg.end();
	vector<Mat>::iterator itbg_mask = maskImg.begin();
	
	for (; itbg < itend; ++itbg, ++itbg_mask)
	{
		maskedImage.push_back(MaskedImage(*itbg, *itbg_mask));
	}
	Mat tempImg;

	/*for (int i = 0; i < mask_st.size(); i++)
		PrintMaskValue(mask_st[i]);*/
	
	/*
	// Show the pyr
	vector<Mat>::iterator itbg = maskImg.begin();
	vector<Mat>::iterator itend = maskImg.end();
	
	int i = 0;
	std::stringstream title;
	Mat tmp;
	for(; itbg < itend; ++itbg){
		title << "Gaussian Pyramid " << i;
		namedWindow(title.str());
		//resize(*itbg, tmp, Size(srcImg.front().cols, srcImg.front().rows));
		//imshow(title.str(), tmp);
		PrintMaskValue(*itbg);
		++i;
		title.clear();
	}
	waitKey();
	*/
}

void Inpaint::Run()
{
	BuildPyr();

	vector<MaskedImage>::iterator maskImageBg = maskedImage.begin();
	vector<MaskedImage>::iterator maskImageEnd = maskedImage.end() - 1;
	//vector<Mat>::iterator srcItBg = srcImg.begin();
	//vector<Mat>::iterator srcItEnd = srcImg.end()-1;
	vector<Mat>::iterator maskItBg = maskImg.begin();
	vector<Mat>::iterator maskItEnd = maskImg.end() - 1;
	//vector<Mask>::iterator maskItBg = mask_st.begin();
	//vector<Mask>::iterator maskItEnd = mask_st.end() - 1;
	vector<Mat>::iterator offsetMapBg_SourceToTarget = offsetMap_SourceToTarget.begin();
	vector<Mat>::iterator offsetMapEnd_SourceToTarget = offsetMap_SourceToTarget.end()-1;
	vector<Mat>::iterator offsetMapBg_TargetToSource = offsetMap_TargetToSource.begin();
	vector<Mat>::iterator offsetMapEnd_TargetToSource = offsetMap_TargetToSource.end()-1;

	Mat tempImg;
	int index = srcImg.size() - 1;
	for (; maskImageEnd >= maskImageBg+1; maskImageEnd--, offsetMapEnd_SourceToTarget--, offsetMapEnd_TargetToSource--, maskItEnd--, index--)
	{
		MaskedImage src = *maskImageEnd;
		
		Mat offset_TargetToSource = *offsetMapEnd_TargetToSource;
		Mat offset_SourceToTarget = *offsetMapEnd_SourceToTarget;

		cout << "Pry " << index << endl;
		if (maskImageEnd == maskedImage.end() - 1)
		{
			// Initialize offsetmap with random values
			target = src.copy();
			for (int i = 0; i < target.row; i++)
				for (int j = 0; j < target.col; j++)
					target.SetMask(i, j, 0);
			
			//PrintMaskValue(target.mask);
			
			RandomizeOffsetMap(src, target, offset_SourceToTarget);
			RandomizeOffsetMap(target, src, offset_TargetToSource);

			//PrintOffsetMap(*(offsetMapEnd_SourceToTarget));
		}
		else
		{	
#ifdef SHOW_INTERMEDIATE
			resize(target.img, tempImg, Size((*maskImageBg).col, (*maskImageBg).row));
			imshow("Pry " + index, tempImg);
			waitKey();
#endif
			//resize(target.img, target.img, Size(src.col, src.row));
			//resize(target.mask, target.mask, Size(src.col, src.row));

			//target.mask = src.mask.clone();
			//target.row = target.mask.rows;
			//target.col = target.mask.cols;

			/*for (int i = 0; i < target.row; i++)
				for (int j = 0; j < target.col; j++)
					target.SetMask(i, j, 0);*/
#ifdef MY_DEBUG
			
			PrintMaskValue(target.mask);
#endif
			
			// Initialize offsetmap with the small offsetmap
			InitOffsetMap(src, target, *(offsetMapEnd_SourceToTarget+1), offset_SourceToTarget);
			InitOffsetMap(target, src, *(offsetMapEnd_TargetToSource+1), offset_TargetToSource);
#ifdef MY_DEBUG
			PrintOffsetMap(*(offsetMapEnd_SourceToTarget));
#endif
		}

		//EM-like
		ExpectationMaximization(src, target, offset_SourceToTarget, offset_TargetToSource, index);
		if (maskImageEnd == maskImageBg)
		{
			cout << "ok in here" << endl;
			break;
		}
	}
	imshow("Pry " + index, target.img);
	cout << "END!!!" << endl;
	waitKey();
}

void Inpaint::RandomizeOffsetMap(const MaskedImage& src, const MaskedImage& target, Mat& offset)
{
#ifdef MY_DEBUG
	//PrintMaskValue(mask);
#endif
	for (int i = 0; i < src.row; i++)
		for (int j = 0; j < src.col; j++)
		{
			//if (150 >= (int)mask.at<uchar>(i, j))
			//if (0 == (int)src.mask.at<uchar>(i, j))
			//{
			//	// Need not search
			//	offset.at<Vec3f>(i, j)[0] = i;
			//	offset.at<Vec3f>(i, j)[1] = j;
			//	offset.at<Vec3f>(i, j)[2] = 0;
			//}
			//else 
			{
				int r_col = rand() % src.col;
				int r_row = rand() % src.row;

				while (1 == (int)src.mask.at<uchar>(r_row, r_col))
				{
					r_col = rand() % src.col;
					r_row = rand() % src.row;
				}

				offset.at<Vec3f>(i, j)[0] = r_row;
				offset.at<Vec3f>(i, j)[1] = r_col;
				offset.at<Vec3f>(i, j)[2] = MaxDis;
			}
		}

	InitOffsetDis(src, target, offset);
}

void Inpaint::InitOffsetMap(const MaskedImage& src, const MaskedImage& target, const Mat& preOff, Mat& offset)
{
	int fx = offset.rows / preOff.rows;
	int fy = offset.cols / preOff.cols;

	fx = fx == 1 ? 2 : fx;
	fy = fy == 1 ? 2 : fy;
	for (int i = 0; i < src.row; i++)
		for (int j = 0; j < src.col; j++)
		{
			//if (!src.IsMasked(i, j))
			//{
			//	// Need not search
			//	offset.at<Vec3f>(i, j)[0] = i;
			//	offset.at<Vec3f>(i, j)[1] = j;
			//	offset.at<Vec3f>(i, j)[2] = 0;
			//}
			//else 
			{
				int xlow = i / fx;
				int ylow = j / fy;
				offset.at<Vec3f>(i, j)[0] = preOff.at<Vec3f>(xlow, ylow)[0] * fx;
				offset.at<Vec3f>(i, j)[1] = preOff.at<Vec3f>(xlow, ylow)[1] * fy;
				offset.at<Vec3f>(i, j)[2] = MaxDis;
			}
		}
	InitOffsetDis(src, target, offset);
}

void Inpaint::InitOffsetDis(const MaskedImage& src, const MaskedImage& target, Mat& offset)
{
	for (int i = 0; i < src.row; i++)
		for (int j = 0; j < src.col; j++)
		{
			//if (!src.mask.at<uchar>(i, j))
			//{
			//	// Need not search
			//	offset.at<Vec3f>(i, j)[0] = i;
			//	offset.at<Vec3f>(i, j)[1] = j;
			//	offset.at<Vec3f>(i, j)[2] = 0;
			//	continue;
			//}

			offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, target, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1]);

			int iter = 0, maxretry = 20;
			while (offset.at<Vec3f>(i, j)[2] == MaxDis && iter < maxretry)
			{
				iter++;
				offset.at<Vec3f>(i, j)[0] = rand() % src.row;
				offset.at<Vec3f>(i, j)[1] = rand() % src.col;
				offset.at<Vec3f>(i, j)[2] = Distance(src, i, j, target, offset.at<Vec3f>(i, j)[0], offset.at<Vec3f>(i, j)[1]);

			}
		}
	//PrintOffsetMap(offset);
}

void Inpaint::ExpectationMaximization(MaskedImage& src, MaskedImage& target, Mat& offset_SourceToTarget, Mat& offset_TargetToSource, int level)
{
	int iterEM = 1 + 2 * (level-1);
	int iterNNF = min(7, 1 + level);

	MaskedImage newtarget;
	for (int i = 0; i < iterEM; i++)
	{
		cout << "ITER " << i << endl;
#ifdef MY_DEBUG
		PrintOffsetMap(offset_SourceToTarget);
#endif
		if (newtarget.row != 0)
		{
			target = newtarget.copy();
			newtarget.row = 0;
		}
		// we force the link between unmasked patch in source/target
		for (int i = 0; i < src.row; i++)
			for (int j = 0; j < src.col; j++)
				if (!src.ContainsMask(i, j))
				{
					offset_SourceToTarget.at<Vec3f>(i, j)[0] = i;
					offset_SourceToTarget.at<Vec3f>(i, j)[1] = j; 
					offset_SourceToTarget.at<Vec3f>(i, j)[2] = 0;

					offset_TargetToSource.at<Vec3f>(i, j)[0] = i;
					offset_TargetToSource.at<Vec3f>(i, j)[1] = j;
					offset_TargetToSource.at<Vec3f>(i, j)[2] = 0;
				}
#ifdef MY_DEBUG
		PrintOffsetMap(offset_SourceToTarget);
#endif
		// PatchMatch
		for (int j = 0; j < iterNNF; j++)
		{
			Iteration(src, target, offset_SourceToTarget, j);
#ifdef MY_DEBUG
			PrintOffsetMap(offset_SourceToTarget);
#endif
			Iteration(target, src, offset_TargetToSource, j);
		}
#ifdef MY_DEBUG
		PrintOffsetMap(offset_SourceToTarget);
#endif
		// Form a target image
		bool upscale = false;
		MaskedImage newSrc;
		if (level > 0 && i == iterEM - 1)
		{
			newSrc = maskedImage[level - 1];
			newtarget = target.Upscale(newSrc.row, newSrc.col);
			upscale = true;
		}
		else
		{
			newSrc = maskedImage[level];
			newtarget = target.copy();
			upscale = false;
		}
		// New a vote array
		Mat vote = Mat(newSrc.img.size(), CV_32FC4, Scalar::all(0));
		VoteForTarget(src, target, offset_SourceToTarget, true, vote, upscale, newSrc);
		VoteForTarget(target, src, offset_TargetToSource, false, vote, upscale, newSrc);
		FormTargetImg(newtarget, vote);
		//DeleteVoteArray(vote);
	}

	target = newtarget.copy();
}

void Inpaint::VoteForTarget(const MaskedImage& src, const MaskedImage& tar, const Mat& offset, bool sourceToTarget, Mat vote, bool upscale, const MaskedImage& newsrc)
{
	//targetImg = src.clone();

	for (int i = 0; i < src.row; i++)
		for (int j = 0; j < src.col; j++)
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

					if (xs < 0 || xs >= src.row) continue;
					if (ys < 0 || ys >= src.col) continue;
					if (xt < 0 || xt >= src.row) continue;
					if (yt < 0 || yt >= src.col) continue;

					if (upscale)
					{
						int nxt = 2 * xt + 1, nxs = 2 * xs + 1, nyt = 2 * yt + 1, nys = 2 * ys + 1;
						if (2 * xt + 1 >= newsrc.row)
						{
							nxt--;
						}
						if (2 * xs + 1 >= newsrc.row)
						{
							nxs--;
						}
						if (2 * ys + 1 >= newsrc.col)
						{
							nys--;
						}
						if (2 * yt + 1 >= newsrc.col)
						{
							nyt--;
						}
						
						{
							WeightedCopy(newsrc, 2 * xs, 2 * ys, vote, 2 * xt, 2 * yt, w);
							//WeightedCopy(newsrc, 2 * xs + 1, 2 * ys, vote, 2 * xt + 1, 2 * yt, w);
							WeightedCopy(newsrc, nxs, 2 * ys, vote, nxt, 2 * yt, w);
							//WeightedCopy(newsrc, 2 * xs, 2 * ys + 1, vote, 2 * xt, 2 * yt + 1, w);
							WeightedCopy(newsrc, 2 * xs, nys, vote, 2 * xt, nyt, w);
							//WeightedCopy(newsrc, 2 * xs + 1, 2 * ys + 1, vote, 2 * xt + 1, 2 * yt + 1, w);
							WeightedCopy(newsrc, nxs, nys, vote, nxt, nyt, w);
						}
						
					}
					else 
						WeightedCopy(newsrc, xs, ys, vote, xt, yt, w);
				}
			}
		}
}

void Inpaint::WeightedCopy(const MaskedImage& src, int xs, int ys, Mat vote, int xd, int yd, double w)
{
	vote.at<Vec4f>(xd, yd)[0] += w * src.img.at<Vec3b>(xs, ys)[0];
	vote.at<Vec4f>(xd, yd)[1] += w * src.img.at<Vec3b>(xs, ys)[1];
	vote.at<Vec4f>(xd, yd)[2] += w * src.img.at<Vec3b>(xs, ys)[2];
	vote.at<Vec4f>(xd, yd)[3] += w;
}

void Inpaint::FormTargetImg(MaskedImage& target, Mat vote)
{
	for (int i = 0; i < target.row; i++)
	{
		for (int j = 0; j < target.col; j++)
		{
			if (vote.at<Vec4f>(i, j)[3] > 0)
			{
				target.img.at<Vec3b>(i, j)[0] = (int)(vote.at<Vec4f>(i, j)[0] / vote.at<Vec4f>(i, j)[3]);
				target.img.at<Vec3b>(i, j)[1] = (int)(vote.at<Vec4f>(i, j)[1] / vote.at<Vec4f>(i, j)[3]);
				target.img.at<Vec3b>(i, j)[2] = (int)(vote.at<Vec4f>(i, j)[2] / vote.at<Vec4f>(i, j)[3]);
				target.SetMask(i, j, 0);
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
	delete[] vote;
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

int Inpaint::Distance(const MaskedImage &Src, int xs, int ys, const MaskedImage &Dst, int xt, int yt)
{
	int dis = 0, wsum = 0, ssdmax = 255 * 255 * 9;

	for (int dy = -(PatchSize / 2); dy <= PatchSize / 2; dy++)
		for (int dx = -(PatchSize / 2); dx <= PatchSize / 2; dx++)
		{
			wsum += ssdmax;

			int xks = xs + dx, yks = ys + dy;
			if (xks < 1 || xks >= Src.row - 1) {dis += ssdmax; continue; }
			if (yks < 1 || yks >= Src.col - 1) {dis += ssdmax; continue; }

			if (Src.IsMasked(xks, yks)) { dis += ssdmax; continue; }

			int xkt = xt + dx, ykt = yt + dy;
			if (xkt < 1 || xkt >= Dst.row - 1) {dis += ssdmax; continue; }
			if (ykt < 1 || ykt >= Dst.col - 1) {dis += ssdmax; continue; }

			if (Dst.IsMasked(xkt, ykt)) { dis += ssdmax; continue; }
			// SSD distance between pixels (each value is in [0,255^2])
			long long ssd = 0;
			for(int band = 0; band < 3; band++) {
				// pixel values
				int s_value = Src.img.at<Vec3b>(xks, yks)[band];
				int t_value = Src.img.at<Vec3b>(xkt, ykt)[band];

				// pixel horizontal gradients (Gx)
				int s_gx = 128 + (Src.img.at<Vec3b>(xks + 1, yks)[band] - Src.img.at<Vec3b>(xks - 1, yks)[band]) / 2;
				int t_gx = 128 + (Dst.img.at<Vec3b>(xkt + 1, ykt)[band] - Dst.img.at<Vec3b>(xkt - 1, ykt)[band]) / 2;

				// pixel vertical gradients (Gy)
				int s_gy = 128 + (Src.img.at<Vec3b>(xks, yks + 1)[band] - Src.img.at<Vec3b>(xks, yks - 1)[band]) / 2;
				int t_gy = 128 + (Dst.img.at<Vec3b>(xkt, ykt + 1)[band] - Dst.img.at<Vec3b>(xkt, ykt - 1)[band]) / 2;

				ssd += (s_value-t_value) * (s_value-t_value); // distance between values in [0,255^2]
				ssd += (s_gx-t_gx) * (s_gx-t_gx); // distance between Gx in [0,255^2]
				ssd += (s_gy-t_gy) * (s_gy-t_gy); // distance between Gy in [0,255^2]
			}

			// add pixel distance to global patch distance
			dis += ssd;
		}

		return (int) ((long long) MaxDis * dis / (long long)wsum);
}
void Inpaint::Iteration(MaskedImage& src, MaskedImage& target, Mat& offset, int iter)
{
	for (int i = 0; i < src.row; i++)
		for (int j = 0; j < src.col; j++)
		{
			if (offset.at<Vec3f>(i, j)[2] > 0)
			{
				Propagation(src, target, offset, i, j, iter);
				RandomSearch(src, target, offset, i, j);
			}
		}
}

void Inpaint::Propagation(const MaskedImage& src, const MaskedImage& tar, Mat& offset, int row, int col, int dir)
{
	int xp, yp, dp;
	dir %= 2;
	if (0 == dir) dir = -1;

	if (col - dir > 0 && col - dir < src.col)
	{
		xp = offset.at<Vec3f>(row, col - dir)[0];
		yp = offset.at<Vec3f>(row, col - dir)[1] + dir;
		dp = Distance(src, row, col, tar, xp, yp);

		if (dp < offset.at<Vec3f>(row, col)[2])
		{
			offset.at<Vec3f>(row, col)[0] = xp;
			offset.at<Vec3f>(row, col)[1] = yp;
			offset.at<Vec3f>(row, col)[2] = dp;
		}
	}

	if (row - dir > 0 && row - dir < src.row)
	{
		xp = offset.at<Vec3f>(row - dir, col)[0] + dir;
		yp = offset.at<Vec3f>(row - dir, col)[1];
		dp = Distance(src, row, col, tar, xp, yp);

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
	//Mat DstPatch = GetPatch(targetImg, row, col);
	//Mat SrcPatch = GetPatch(src, offset.at<Vec3f>(row, col)[0], offset.at<Vec3f>(row, col)[1]);
	//Mat LeftPatch, RightPatch, UpPatch, DownPatch;

	//if (0 == dir % 2)
	//{
	//	if (col - 1 >= 0)
	//		LeftPatch = GetPatch(src, offset.at<Vec3f>(row, col - 1)[0], offset.at<Vec3f>(row, col - 1)[1] + 1);
	//	if (row - 1 >= 0)
	//		UpPatch = GetPatch(src, offset.at<Vec3f>(row - 1, col)[0] + 1, offset.at<Vec3f>(row - 1, col)[1] + 1);

	//	int location = GetMinPatch(DstPatch, SrcPatch, LeftPatch, UpPatch);

	//	switch (location)
	//	{
	//	case 2:
	//		offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row, col - 1)[0];
	//		offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row, col - 1)[1] + 1;
	//		//offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
	//		break;
	//	case 3:
	//		offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row - 1, col)[0] + 1;
	//		offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row - 1, col)[1];
	//		//offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
	//		break;
	//	}
	//}
	//else 
	//{
	//	if (col + 1 < src.cols)
	//		RightPatch = GetPatch(src, offset.at<Vec3f>(row, col + 1)[0], offset.at<Vec3f>(row, col + 1)[1] - 1);
	//	if (row + 1 < src.rows)
	//		DownPatch = GetPatch(src, offset.at<Vec3f>(row + 1, col)[0] - 1, offset.at<Vec3f>(row + 1, col)[1] - 1);

	//	int location = GetMinPatch(DstPatch, SrcPatch, RightPatch, DownPatch);

	//	switch (location)
	//	{
	//	case 2:
	//		offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row, col + 1)[0];
	//		offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row, col + 1)[1] - 1;
	//		//offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
	//		break;
	//	case 3:
	//		offset.at < Vec3f > (row, col)[0] = offset.at < Vec3f > (row + 1, col)[0] - 1;
	//		offset.at < Vec3f > (row, col)[1] = offset.at < Vec3f > (row + 1, col)[1];
	//		//offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
	//		break;
	//	}
	//}
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
	//Mat DstPatch = GetPatch(targetImg, row, col);
	//Mat SrcPatch = GetPatch(src, offset.at<Vec3f>(row, col)[0], offset.at<Vec3f>(row, col)[1]);

	//int w = min(src.cols, src.rows);
	//
	//while (w > 0)
	//{
	//	int x = rand() % w;
	//	int y = rand() % w;

	//	Mat candidate = GetPatch(src, x, y);

	//	int dis1 = Distance(SrcPatch, DstPatch);
	//	int dis2 = Distance(SrcPatch, candidate);

	//	if (dis2 < dis1)
	//	{
	//		offset.at < Vec3f > (row, col)[0] = x;
	//		offset.at < Vec3f > (row, col)[1] = y;
	//		//offset.at < Vec3f > (row, col)[2] = Distance(src, row, col, targetImg, offset.at < Vec3f > (row, col)[0], offset.at < Vec3f > (row, col)[1], mask);
	//	}
	//	w /= 2;
	//}
}

void Inpaint::RandomSearch(const MaskedImage& src, const MaskedImage& tar, Mat& offset, int row, int col)
{
	int w = max(src.col, src.row);
	
	while (w > 0)
	{
		int x = rand() % (2 * w) - w + row;
		int y = rand() % (2 * w) - w + col;
		x = max(0, min(src.row - 1, x));
		y = max(0, min(src.col - 1, y));

		int d = Distance(src, row, col, tar, x, y);

		if (d < offset.at < Vec3f > (row, col)[2])
		{
			offset.at < Vec3f > (row, col)[0] = x;
			offset.at < Vec3f > (row, col)[1] = y;
			offset.at < Vec3f > (row, col)[2] = d;
		}
		w /= 2;
	}
}