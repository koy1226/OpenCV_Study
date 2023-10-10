#pragma once
#include "../Common/Common.h"

int minScratchSize = 10;
int maxScratchSize = 500;

int main() {
	// Step 1: Read image names
	vector<String> filenames;
	vector<String> noLens_filenames;
	string filePath = "../camera_module/*.jpg";
	glob(filePath, filenames);

	for (const auto& filename : filenames) {
		Mat img = imread(filename, IMREAD_COLOR);

		// Step 2: Check if lens exists
		cv::Vec3b pixel = img.at<cv::Vec3b>(500, 1260);
		if (pixel[0] < 85) {
			noLens_filenames.push_back(filename);
			continue;
		}

		// Step 3: Divide the image into four
		int width = img.cols;
		int height = img.rows;
		const int height_quarter = height / 4;
		vector<Mat> sub_imgs;
		Mat obj_img;
		Rect roi2 = Rect(1200, 400, 3400 - 1200, 2600 - 400);
		for (int i = 0; i < 4; ++i) {
			Rect roi(0, i * height_quarter, width, height_quarter);
			obj_img = img(roi).clone();
			sub_imgs.push_back(img(roi2).clone());
		}

		// Step 4~6: Perform defect detection for each image
		for (Mat& sub_img : sub_imgs) {
			// Step 4: Remove unnecessary objects
			// Convert to grayscale and apply threshold to isolate the lens
			//sub_img = ~sub_img;
			Mat gray;
			cvtColor(sub_img, gray, COLOR_BGR2GRAY);
			//gray = ~gray;
			//threshold(gray, gray, 50, 255, THRESH_BINARY_INV);

			// Step 5: Detect defects (white spots)
			// Apply another threshold to detect white spots
			Mat defects;
			threshold(gray, defects, 70, 255, THRESH_BINARY);
			//Mat defects2;
			//threshold(gray, defects1, 90, 255, THRESH_BINARY);

			// Step 6: Mark defects and save results
			 // Draw a circle around every white spot detected
			vector<vector<Point>> contours;
			findContours(defects, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			sort(contours.begin(), contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
				return a.size() > b.size(); });
			// Create the masks using the two largest contours
			std::vector<Point>& largest_contour = contours[0];
			std::vector<Point>& second_largest_contour = contours[1];

			Mat mask_large = Mat::zeros(defects.size(), CV_8UC3);
			Mat mask_small = Mat::zeros(defects.size(), CV_8UC3);

			Point2f center1, center2;
			float radius1 = 0.0, radius2 = 0.0;
			minEnclosingCircle(largest_contour, center1, radius1);
			minEnclosingCircle(second_largest_contour, center2, radius2);
			radius1 = radius1 - 20.0;
			//radius2 = radius2 - 5.0;
			// Now you can draw the circles as before
			circle(mask_large, center1, radius1, Scalar(255, 255, 255), LineTypes::FILLED);
			circle(mask_small, center2, radius2, Scalar(255, 255, 255), LineTypes::FILLED);

			Mat mask_donut = mask_large - mask_small;

			Mat reduced_obj_img = sub_img & mask_donut;
			Mat reduced_gray;
			cvtColor(reduced_obj_img, reduced_gray, COLOR_BGR2GRAY);
			//reduced_gray = sub_img & mask_donut;
			//reduced_gray = ~reduced_gray;
			Mat reduced_defects;
			threshold(reduced_gray, reduced_defects, 200, 255, THRESH_BINARY);

			vector<vector<Point>> new_contours;
			findContours(reduced_defects, new_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			sort(new_contours.begin(), new_contours.end(), [](const vector<Point>& a, const vector<Point>& b) {
				return a.size() > b.size(); });

			int index = 1;
			for (const auto& contour : new_contours) {
				if (contour.size() > maxScratchSize || contour.size() < minScratchSize)
					continue;
				Rect r = boundingRect(contour);
				// Check if it's possible to inflate the rectangle without going out of bounds
				int rx_point = 0, ry_point = 0;
				int inflation = 25;
				if (r.x - inflation >= 0 && r.y - inflation >= 0 &&
					r.width + 2 * inflation < img.cols && r.height + 2 * inflation < img.rows) {
					r.x -= inflation;
					r.y -= inflation;
					r.width += 2 * inflation;
					r.height += 2 * inflation;
					rx_point = r.x + r.width / 2;
					ry_point = r.y + r.height / 2;
				}
				// Reduced image to Original image
				int rx_point_orig = rx_point + roi2.x;
				int ry_point_orig = ry_point + roi2.y;

				// Draw the circle on the original image
				circle(reduced_obj_img, Point(rx_point, ry_point), max(r.width, r.height) / 2, Scalar(0, 255, 0), 3);
				circle(sub_img, Point(rx_point, ry_point), max(r.width, r.height) / 2, Scalar(0, 255, 0), 2);
				circle(obj_img, Point(rx_point_orig, ry_point_orig), max(r.width, r.height) / 2, Scalar(0, 255, 0), 2);
				
				String index_str = std::format("Defect[{}]", index);
				int font_face = FONT_HERSHEY_SIMPLEX;
				double font_scale = 1;
				int thickness = 2;
				int baseline = 0;
				Size text_size = getTextSize(index_str, font_face, font_scale, thickness, &baseline);

				// calculate text position so that text is centered
				Point text_org((rx_point_orig - text_size.width / 2), (ry_point_orig - (r.height / 2 + text_size.height / 2)));

				putText(obj_img, index_str, text_org, font_face, font_scale, Scalar(0, 255, 255), thickness);

				index++;
			}
			cout << "check! " << endl;

		}
	}
	return 0;
}

/*
// Convert to grayscale
		cv::Mat gray_img;
		cv::cvtColor(sub_img, gray_img, COLOR_BGR2GRAY);
		//gray_img = ~gray_img;

		// Binary thresholding
		cv::Mat binary;
		cv::threshold(gray_img, binary, 127, 255,
			cv::THRESH_BINARY);  // ThresholdTypes::

		// Find contours
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(binary, contours, hierarchy, cv::RETR_TREE,
			cv::CHAIN_APPROX_SIMPLE);

		// Draw contours
		RNG rng(12345);
		cv::Mat contourImage = cv::Mat::zeros(gray_img.size(), CV_8UC1);
		Mat contourImage;
		cvtColor(binary, contourImage, COLOR_GRAY2BGR);

		for (size_t k = 0; k < contours.size(); k++) {
			//cv::drawContours(contourImage, contours, (int)k, Scalar(0, 0, 0), 2, LINE_8, hierarchy, 0);

			// Generate random color
			Scalar color =
				Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			// Draw contours
			drawContours(contourImage, contours, (int)k, color, 2, LINE_8, hierarchy,
				0);

			// Draw minimum enclosing circle
			Point2f center(0, 0);
			float radius = 0;
			cv::minEnclosingCircle(contours[k], center, radius);
			cv::circle(contourImage, center, cvRound(radius), Scalar(255, 255, 255),2);

			// Draw markers
			cv::drawMarker(contourImage, center, Scalar(0, 255, 255), MarkerTypes::MARKER_STAR);

			cv::namedWindow("img", WINDOW_AUTOSIZE);
			imshow("img", binary);
			cv::namedWindow("contour", WINDOW_AUTOSIZE);
			imshow("contour",contourImage);
			cv::waitKey(0);
		}
// Save feature's value
vector<string> vDesc;

// Draw contours
RNG rng(12345);
cv::Mat contourImage = cv::Mat::zeros(gray_img.size(), CV_8UC3);

for (size_t i = 0; i < contours.size(); i++) {
	// For �ﰢ�� ������ ������ contour X
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

*/

