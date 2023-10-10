#pragma once
#include "../Common/Common.h"
/*
Pattern�� ������ ���Ͻÿ�. N�� ������Ʈ�� �����Ͻÿ�.
*/
int main() {
    // Step 1: Read image names
    //vector<String> filenames;
    //glob(filePath, filenames);
    string filePath = "/images/";
    string fileName = filePath + "wafer_dies.png";
    Mat gray_img = cv::imread(fileName, cv::ImreadModes::IMREAD_ANYCOLOR);

	//Mat gray_img;
	//cvtColor(Image, gray_img, COLOR_BGR2GRAY);

	Mat draw_color = gray_img.clone();
	cvtColor(gray_img, draw_color, ColorConversionCodes::COLOR_GRAY2BGR);

	//to do 
	int width = gray_img.cols;
	int height = gray_img.rows;
	int channel = gray_img.channels();

	int ptrn_w = 160;
	int ptrn_h = 84;
	Mat ptrn_img = gray_img(Rect(0, 0, ptrn_w, ptrn_h)).clone();
	//find same positions

	Mat find_img = Mat::zeros(Size(gray_img.cols - ptrn_img.cols+1, gray_img.rows - ptrn_img.rows+1), CV_32FC1);
	//image whole
	vector<float> scores;
	vector<Point> locations;
	for (int row = 0; row < gray_img.rows - ptrn_img.rows+1; row++)
	{
		for (int col = 0; col < gray_img.cols - ptrn_img.cols+1; col++)
		{

			//compare input img and ptrn img
			float find_value = 0;
			for (int y = 0; y < ptrn_img.rows; y++)
			{
				for (int x = 0; x < ptrn_img.cols; x++)
				{
					uchar p1 = gray_img.data[(row + y) * gray_img.cols + (col + x)];
					uchar p2 = ptrn_img.data[(y)*ptrn_img.cols + (x)];
					float tmp = p2 - p1;
					tmp = std::abs<float>(tmp);
					find_value += tmp;
				}
			}
			size_t len = ptrn_img.total();
			find_value /= len;
			find_img.at<float>(row, col) = find_value;

			if (find_value < 10)
			{
				scores.push_back(find_value);
				locations.push_back(Point(col, row));
			}
		}
	}

	RNG rng(12345);
	for (const auto& location : locations)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		cv::rectangle(draw_color, Rect(location.x, location.y, ptrn_w, ptrn_h), color, 1);
	}
	for (const auto& location : locations)
	{
		cv::drawMarker(draw_color, location, Scalar(0, 255, 255), MarkerTypes::MARKER_CROSS);
	}
	int a = 0;

return 1;
}

/* References

Mat rgn_holes;
double thres_min = 250;
threshold(gray_img, rgn_holes, thres_min, 255, ThresholdTypes::THRESH_BINARY);


std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;
cv::findContours(rgn_holes, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

RNG rng(12345);
size_t cnt = 0;
for (size_t i = 0; i < contours.size(); i++)
{
	if (contours[i].size() < 5) continue;
	Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	cv::Rect rt = cv::boundingRect(contours[i]);
	cv::RotatedRect rrt = cv::fitEllipse(contours[i]);
	cv::rectangle(draw_color, rt, color, 1);
	drawContours(draw_color, contours, (int)i, color, 2, LINE_8, hierarchy, 0);

	size_t Len1 = std::max(rrt.boundingRect().width, rrt.boundingRect().height);
	size_t Len2 = std::min(rrt.boundingRect().width, rrt.boundingRect().height);
	float diameter = (Len1 + Len2) / 2;
	string msg = std::format("ID[{}] - D:{}", ++cnt, diameter);
	putText(draw_color, msg, Point(rt.x + rt.width / 2, rt.y + rt.height / 2),
		FONT_HERSHEY_SIMPLEX, 1.0,color,2);
}

*/
	