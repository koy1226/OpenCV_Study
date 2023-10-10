#pragma once
#include "../Common/Common.h"

namespace fs = std::filesystem;

int main() {

	string fileDir = "/Willie/";
	string search_file = "Willie.jpg";
	string pattern_file = "me.jpg";

	const string& filePath = fileDir + search_file;
	Mat seach_img = cv::imread(filePath, cv::ImreadModes::IMREAD_ANYCOLOR);
	Mat draw_color = seach_img.clone();

	const string& filePath_ptrn = fileDir + pattern_file;
	Mat pattern_img = cv::imread(filePath_ptrn, cv::ImreadModes::IMREAD_ANYCOLOR);

	Mat search_gray_img, pattern_gray_img;
	cvtColor(seach_img, search_gray_img, ColorConversionCodes::COLOR_BGR2GRAY);
	cvtColor(pattern_img, pattern_gray_img, ColorConversionCodes::COLOR_BGR2GRAY);

	Mat result;
	matchTemplate(search_gray_img, pattern_gray_img, result, cv::TM_CCOEFF_NORMED);

	// find max value after normalization of result
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	double threshold_matching = 0.8;
	Mat result_areas;
	threshold(result, result_areas, threshold_matching, 255, ThresholdTypes::THRESH_BINARY);
	result_areas.convertTo(result_areas, CV_8UC1);
	vector<vector<Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(result_areas, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	RNG rng(12345);
	for (int i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect rt = cv::boundingRect(contours[i]);
		drawMarker(draw_color, rt.tl(), Scalar(255,255,255), MarkerTypes::MARKER_CROSS);
		rectangle(draw_color, Rect(rt.x, rt.y, pattern_gray_img.cols, pattern_gray_img.rows), color, 1);

	}

	return 1;
}