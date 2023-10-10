#pragma once
#include "../Common/Common.h"

const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void*)
{
	low_H = min(high_H - 1, low_H);
	setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
	high_H = max(high_H, low_H + 1);
	setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
	low_S = min(high_S - 1, low_S);
	setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
	high_S = max(high_S, low_S + 1);
	setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
	low_V = min(high_V - 1, low_V);
	setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
	high_V = max(high_V, low_V + 1);
	setTrackbarPos("High V", window_detection_name, high_V);
}

int main1(int argc, char* argv[])
{

	cv::namedWindow(window_capture_name, WINDOW_NORMAL);
	cv::namedWindow(window_detection_name, WINDOW_NORMAL);
	// Trackbars to set thresholds for HSV values
	createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
	createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
	createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
	createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
	createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
	createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
	Mat frame, frame_HSV, frame_threshold;
	string fileDir = "../thirdparty/opencv_470/sources/samples/data/";
	string fileName = fileDir + "find_google_area.png";
	frame = cv::imread(fileName, cv::ImreadModes::IMREAD_COLOR);
	while (true) {

		if (frame.empty())
		{
			break;
		}
		// Convert from BGR to HSV colorspace
		cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
		// Detect the object based on HSV Range Values
		inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
		// Show the frames
		imshow(window_capture_name, frame);
		imshow(window_detection_name, frame_threshold);
		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}
	return 0;
}

int main() {
	// image path
	string filePath = "/image/";
	string fileName = filePath + "find_google_area.png";
	// Load image
	Mat color_img = cv::imread(fileName, cv::ImreadModes::IMREAD_COLOR);
	if (color_img.empty()) {
		std::cout << "Could not open or find the image!\n";
		return -1;
	}
	// Convert to HSV color space
	Mat hsv;
	cvtColor(color_img, hsv, COLOR_BGR2HSV);

	// Define range for red color and create a mask
	Mat mask;
	inRange(hsv, Scalar(167, 26, 0), Scalar(180, 255, 255), mask);//167 26 0 vs 161 140 115
	//175 0 0�ϰ� ��â ħ�� ���ϸ� angle 85�� ����.
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(mask, mask, element);
	cv::dilate(mask, mask, element);
	cv::dilate(mask, mask, element);
	cv::erode(mask, mask, element);
	cv::erode(mask, mask, element);
	cv::erode(mask, mask, element);

	// Find contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Draw contours
	//Mat contourImg = Mat::zeros(color_img.size(), CV_8UC3);
	Mat contourImg = color_img.clone();
	Mat RectImg = color_img.clone();
	
	double largestArea = 0;

	for (size_t i = 0; i < contours.size(); i++) {
		if (contours[i].size() < 85)
			continue;

		drawContours(contourImg, contours, int(i), Scalar(0, 0, 255), 2, LINE_8, hierarchy, 0);

		// Calculate features for the largest contour
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = area;
		}
		double length = arcLength(contours[i], true);
		Moments M = moments(contours[i]);
		int cX = int(M.m10 / M.m00);
		int cY = int(M.m01 / M.m00);
		RotatedRect r_rt = minAreaRect(contours[i]);
		float major_len = r_rt.size.width;
		float minor_len = r_rt.size.height;
		RotatedRect ellipse = fitEllipse(contours[i]);
		float angle = ellipse.angle;
		angle = angle - 90;
		float ratio = (float)r_rt.size.width / r_rt.size.height;

		// Find center and radius
		Point2f center(0, 0);
		float radius = 0;
		cv::minEnclosingCircle(contours[i], center, radius);

		// Draw rotated rectangle
		Point2f vertices[4];
		ellipse.points(vertices);
		for (int i = 0; i < 4; i++)
			line(RectImg, vertices[i], vertices[(i + 1) % 4], Scalar(159, 255, 0), 2);

		cv::ellipse(RectImg, ellipse, Scalar(10, 10, 10));
		// Compute grayscale average within the largest contour
		Mat gray;
		cvtColor(color_img, gray, COLOR_BGR2GRAY);
		Mat mask2 = Mat::zeros(gray.size(), CV_8UC1);
		drawContours(mask2, contours, -1, 255, FILLED); // Draw all contours in mask2
		Scalar mean = cv::mean(gray, mask2);

		// Print features
		std::cout << "======================================" << std::endl;
		std::cout << "Area: " << largestArea << std::endl;
		std::cout << "Length: " << length << std::endl;
		std::cout << "Location: (" << cX << ", " << cY << ")" << std::endl;
		std::cout << "Min radius: " << radius << std::endl;
		std::cout << "Major Length: " << major_len << std::endl;
		std::cout << "Minor Length: " << minor_len << std::endl;
		std::cout << "Angle: " << angle << std::endl;
		std::cout << "Ratio: " << ratio << std::endl;
		std::cout << "Brightness: " << mean.val[0] << std::endl;
		std::cout << "======================================" << std::endl;

		putText(RectImg, std::format("Contour #{}", 1),
			Point(r_rt.boundingRect().tl().x, r_rt.boundingRect().tl().y), 1, 1,
			Scalar(0, 125, 255), 1, 8);
	}

	// show text result
	const char* Result_window = "Result image";
	cv::namedWindow(Result_window);
	cv::imshow(Result_window, contourImg);
	cv::waitKey(0);

	// Save result
	//imwrite("result.bmp", RectImg);

	return 0;
}