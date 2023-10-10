#pragma once
#include "../Common/Common.h"

int main() {
	// image path
	string filePath = "../thirdparty/opencv_470/sources/samples/data/";
	string fileName = filePath + "Object.jpg";
	// Load image
	Mat color_img = cv::imread(fileName, cv::ImreadModes::IMREAD_COLOR);
	if (color_img.empty()) {
		std::cout << "Could not open or find the image!\n";
		return -1;
	}

	// Convert to grayscale and inverse
	cv::Mat gray_img;
	cv::cvtColor(color_img, gray_img, COLOR_BGR2GRAY);
	gray_img = ~gray_img;

	// Binary thresholding
	cv::Mat binary;
	cv::threshold(gray_img, binary, 20, 255,
		cv::THRESH_BINARY);  // ThresholdTypes::

	// Find contours
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL,
		cv::CHAIN_APPROX_SIMPLE);

	// Save feature's value
	vector<string> vDesc;

	// Draw contours
	RNG rng(12345);
	cv::Mat contourImage = cv::Mat::zeros(gray_img.size(), CV_8UC3);

	for (size_t i = 0; i < contours.size(); i++) {
		// For 삼각형 이하의 도형은 contour X
		if (contours[i].size() < 3) continue;

		// Generate random color
		Scalar color =
			Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

		// Draw contours
		drawContours(contourImage, contours, (int)i, color, 2, LINE_8, hierarchy,
			0);

		// Calculate geometric features
		double length = cv::arcLength(contours[i], false);
		double area = cv::contourArea(contours[i]);
		RotatedRect r_rt = cv::minAreaRect(contours[i]);
		rectangle(contourImage, r_rt.boundingRect(), Scalar(128, 128, 255));

		// Draw minimum enclosing circle
		Point2f center(0, 0);
		float radius = 0;
		cv::minEnclosingCircle(contours[i], center, radius);
		cv::circle(contourImage, center, radius, Scalar(255, 255, 255));

		// Draw markers
		cv::drawMarker(contourImage, center, Scalar(0, 255, 255),
			MarkerTypes::MARKER_STAR);
		// cv::drawMarker(contourImage, center, Scalar(128, 255, 255),
		// MarkerTypes::MARKER_TILTED_CROSS);

		// Calculate mean grayscale value
		cv::Mat labels = cv::Mat::zeros(binary.size(), CV_8UC1);
		drawContours(labels, contours, i, color, cv::FILLED);
		Rect roi = cv::boundingRect(contours[i]);
		Scalar mean = cv::mean(gray_img(roi), labels(roi) == i);

		// Open Contour's window
		Mat finalImage = Mat::zeros(contourImage.size(), contourImage.type());

		for (size_t i = 0; i < contours.size(); i++) {
			labels = 0;
			// Fill in the current contour
			drawContours(labels, contours, i, Scalar(255), FILLED);

			// Convert grayscale mask to color
			Mat mask_color;
			cvtColor(labels, mask_color, COLOR_GRAY2BGR);

			// Bitwise-AND with original color image
			Mat reduced_color = contourImage & mask_color;

			// Add to the final image
			finalImage = finalImage + reduced_color;
		}

		cv::namedWindow("All Contours", WINDOW_AUTOSIZE);
		cv::imshow("All Contours", finalImage);
		cv::waitKey(0);

		RotatedRect ellipse;
		ellipse = cv::minAreaRect(contours[i]);
		cv::ellipse(contourImage, ellipse, Scalar(10, 10, 10));

		// Draw rotated rectangle
		const int rect_poly_vertexs = 4;
		cv::Point2f vertices2f[rect_poly_vertexs];
		r_rt.points(vertices2f);
		cv::Point vertices[rect_poly_vertexs];
		for (int i = 0; i < rect_poly_vertexs; ++i) {
			vertices[i] = vertices2f[i];
		}
		for (int i = 0; i < 4; i++)
			line(contourImage, vertices[i], vertices[(i + 1) % 4],
				Scalar(128, 128, 255), 2);

		string desc = "";
		desc += std::format("Contour #{}\n", i + 1);
		desc += std::format("Length {}\n", length);
		desc += std::format("Area {}\n", area);
		desc += std::format("Min radius {}\n", radius);
		desc += std::format("major len {}\n",
			std::max(ellipse.boundingRect().size().width,
				ellipse.boundingRect().size().height));
		desc += std::format("minor len {}\n",
			std::min(ellipse.boundingRect().size().width,
				ellipse.boundingRect().size().height));
		desc +=
			std::format("ratio {}\n", ellipse.boundingRect().size().aspectRatio());
		desc += std::format("Gray mean {}\n", mean[0]);
		desc += std::format("x {}\n", r_rt.boundingRect().tl().x);
		desc += std::format("y {}\n", r_rt.boundingRect().tl().y);
		putText(contourImage, std::format("Contour #{}\n", i + 1),
			Point(r_rt.boundingRect().tl().x, r_rt.boundingRect().tl().y), 1, 1,
			Scalar(255, 255, 0), 1, 8);
		cout << desc << endl;
	}
	// save image
	imwrite("result.bmp", contourImage);

	// show text result
	const char* draw_window = "Draw image";
	cv::namedWindow(draw_window);
	cv::imshow(draw_window, contourImage);
	cv::waitKey(0);



	return 0;
}

/*
// fitEllipse 함수의 기본 요구 사항: 최소한 5개 이상의 점이 필요
if (contours[i].size() > 5)
{
		ellipse = cv::fitEllipse(contours[i]);
		cv::ellipse(contourImage, ellipse, Scalar(10, 10, 10));
}
*/
// Convert to grayscale1 => 노란색을 못찾거나 윤곽선이 안이쁨
// cv::Mat gray_img;
// cv::cvtColor(color_img, gray_img, COLOR_BGR2GRAY);
// Binary thresholding
// cv::Mat binary;
// cv::threshold(gray_img, binary, 200, 255, cv::THRESH_BINARY_INV);
// //ThresholdTypes::
// Convert to grayscale => 노란색을 못찾거나 윤곽선이 안이쁨
// cv::threshold(gray_img, binary, 0, 255, THRESH_BINARY_INV |
// THRESH_TRIANGLE);// adaptiveThreshold(gray_img, binary, 255,
// ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

// Mat contourImage = Mat::zeros(color_img.size(), color_img.type());
// for (size_t i = 0; i < contours.size(); i++) {
//	// Find average color of the shape in the original image
//	Vec4d color_avg(0, 0, 0, 0);
//	for (const auto& p : contours[i]) {
//		// Get the color of the pixel at the contour point in the original
//image 		Vec3b color = color_img.at<Vec3b>(p);
//		// Accumulate color values
//		color_avg += Vec4d(color[0], color[1], color[2], 0);
//	}
//	// Calculate average color
//	color_avg /= static_cast<double>(contours[i].size());
//	// Draw the contour with the average color
//	Scalar color_avg_scalar(color_avg[0], color_avg[1], color_avg[2],
//color_avg[3]); 	drawContours(contourImage, contours, i, color_avg_scalar, 1,
//LINE_8, hierarchy, 0);
// }
double largestArea = 0;
vector<Point> largestContour;

for (size_t i = 0; i < contours.size(); i++) {
	// Calculate area
	double area = contourArea(contours[i]);
	// If the area of the current contour is greater than the area of the previously stored largest contour
	if (area > largestArea) {
		largestArea = area;
		largestContour = contours[i];
	}
}

// Now, largestContour stores the largest contour and largestArea has the area of the largest contour.

// Calculate features for the largest contour
double length = arcLength(largestContour, true);
Moments M = moments(largestContour);
int cX = int(M.m10 / M.m00);
int cY = int(M.m01 / M.m00);
RotatedRect r_rt = minAreaRect(largestContour);
float angle = r_rt.angle;
float ratio = (float)r_rt.size.width / r_rt.size.height;

// Draw rotated rectangle
Point2f vertices[4];
r_rt.points(vertices);
for (int i = 0; i < 4; i++)
	line(contourImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));

// Compute grayscale average within the largest contour
Mat gray;
cvtColor(color_img, gray, COLOR_BGR2GRAY);
Mat mask2 = Mat::zeros(gray.size(), CV_8UC1);
drawContours(mask2, contours, -1, 255, FILLED); // Draw all contours in mask2
Scalar mean = cv::mean(gray, mask2);

// Create text with feature values
string text = format("Area: %.2f, Length: %.2f, Angle: %.2f, Ratio: %.2f, Brightness: %.2f", largestArea, length, angle, ratio, mean.val[0]);

// Put text
Point text_position = Point(cX, cY);
putText(contourImg, text, text_position, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
