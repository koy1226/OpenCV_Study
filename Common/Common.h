#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <iomanip>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef OPENCV_470
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/type_c.h>
#endif //OPENCV_470

#ifdef _DEBUG
#pragma comment(lib, "opencv_world470d.lib")
#else //RELEASE
#pragma comment(lib, "opencv_world470d.lib")
#endif
using namespace std;
using namespace cv;
