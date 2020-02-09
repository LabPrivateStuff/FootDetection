#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;
using namespace cv;

// 比较两个
bool compareContourX(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	int x1 = (cv::boundingRect(cv::Mat(contour1)).br()+ cv::boundingRect(cv::Mat(contour1)).tl()).x/2;
	int x2 = (cv::boundingRect(cv::Mat(contour2)).br() + cv::boundingRect(cv::Mat(contour2)).tl()).x / 2;
	return (x1 < x2); // 左的在前
}
class PressureData
{
public:
	double Mx, My, force, Fx, Fy;
	vector<vector<cv::Point>> contours;
	PressureData();
	PressureData(double Mx, double My, double force);
	PressureData& operator=(const PressureData& pData)
	{
		// 处理自我赋值
		if (this == &pData) return *this;
		Mx = pData.Mx;
		My = pData.My;
		force = pData.force;
		Fx = pData.Fx;
		Fy = pData.Fy;
		// 处理链式赋值
		return *this;
	}
	friend void swap(PressureData& p1, PressureData& p2)
	{
		swap(p1.Mx, p2.Mx);
		swap(p1.My, p2.My);
		swap(p1.force, p2.force);
		swap(p1.Fx, p2.Fx);
		swap(p1.Fy, p2.Fy);
		swap(p1.contours, p2.contours);
	}
	void printData()
	{
		printf("Mx: %f\tMy: %f\t Force: %f\tFx: %f\tFy: %f\n", Mx, My, force, Fx, Fy);
	}
	void clear()
	{
		Mx = My = force = Fx = Fy = -1;
		contours.clear();
	}
};
PressureData::PressureData()
{
	Mx = My = force = Fx = Fy = -1;
	
}
PressureData::PressureData(double _Mx, double _My, double _force)
{
	Mx = _Mx;
	My = _My;
	force = _force;
	Fx = Mx / _force;
	Fy = My / _force;
}
class DataSaver
{
public:
	ofstream file;
	DataSaver()
	{
		file.open("data.csv");
		file<<"Left force,Left X,Left Y,Right force,Right X,Right Y\n";
	}
	~DataSaver()
	{
		file.close();
	}
	void save(const PressureData& lData, const PressureData& rData)
	{
		file<<lData.force<<","<<lData.Fx<<","<<lData.Fy<<",";
		file<<rData.force<<","<<rData.Fx<<","<<rData.Fy<<"\n";
	}
};

int main()
{
	VideoCapture cap("demo.avi");
	DataSaver saver;
	// VideoWriter writer("result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(57, 129));
	Mat frame;
	int delay = 1000/30;
	namedWindow("Frame", cv::WINDOW_NORMAL);
	// namedWindow("Binary", cv::WINDOW_NORMAL);
	// namedWindow("leftMask", cv::WINDOW_NORMAL);
	// namedWindow("rightMask", cv::WINDOW_NORMAL);
	// namedWindow("left", cv::WINDOW_NORMAL);
	// namedWindow("right", cv::WINDOW_NORMAL);
	
	Mat dilateEle = getStructuringElement(cv::MORPH_DILATE, cv::Size(5, 19));
	PressureData lData;
	PressureData rData;
	while (true)
	{
		auto start = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if (frame.empty()) break;
		Mat grey;
		// frame *= 10;
		copyMakeBorder(frame, frame, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
		if (frame.channels() == 3) cvtColor(frame, grey, COLOR_BGR2GRAY);
		Mat binary = grey.clone();
		threshold(binary, binary, 0, 255, THRESH_BINARY);
		dilate(binary, binary, dilateEle, Point(-1,-1), 1);
		vector<vector<Point2i>> contours, leftContour, rightContour;
		findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		assert (contours.size()<3); // serious error, 之后加入报错机制

		if (contours.size()<1)
		{
			cout << "contours number invalid" << endl;
			// writer << frame;
			imshow("Frame", frame);
			// waitKey(delay);
			saver.save(lData, rData);
			continue;
		}
		
		Mat leftMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat rightMask = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat left = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		Mat right = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
		
		if (contours.size() == 1)
		{
			Rect r = boundingRect(Mat(contours[0]));
			Point br = (r.br()+r.tl())/2;
			if (frame.cols/2 < br.x)
			{
				drawContours(leftMask, contours, -1, Scalar(255), -1); 
				grey.copyTo(left, leftMask);
				Moments lM = moments(left);
				lData = PressureData(lM.m10, lM.m01, sum(left)[0]);
				lData.contours = contours;
				if (lData.Fx < frame.cols/2) swap(lData, rData);
			}
			else
			{
				drawContours(rightMask, contours, -1, Scalar(255), -1);
				grey.copyTo(right, rightMask);
				Moments rM = moments(right);
				rData = PressureData(rM.m10, rM.m01, sum(right)[0]);
				rData.contours = contours;
				if (frame.cols / 2 < rData.Fx) swap(lData, rData);
			}
		}
		else if(contours.size() == 2)
		{
			// std::sort(contours.begin(), contours.end(), compareContourX);
			leftContour.push_back(contours[1]);
			rightContour.push_back(contours[0]);
			drawContours(leftMask, leftContour, -1, Scalar(255), -1);
			drawContours(rightMask, rightContour, -1, Scalar(255),-1);
			grey.copyTo(left, leftMask);
			grey.copyTo(right, rightMask);
			Moments lM = moments(left);
			lData = PressureData(lM.m10, lM.m01, sum(left)[0]);
			Moments rM = moments(right);
			rData = PressureData(rM.m10, rM.m01, sum(right)[0]);
			lData.contours = leftContour;
			rData.contours = rightContour;
			if (lData.Fx < rData.Fx) swap(lData, rData);
				
		}
		if (0 < lData.force)
		{
			rectangle(frame, boundingRect(lData.contours[0]), Scalar(0, 0, 255));
		}
		if (0 < rData.force)
		{
			rectangle(frame, boundingRect(rData.contours[0]), Scalar(255, 0, 0));
		}
		cout << "Left" << endl;
		lData.printData();
		cout << "Right" << endl;
		rData.printData();
		imshow("Frame", frame);
		// writer << frame;
		// imshow("Binary", binary);
		// imshow("leftMask", leftMask);
		// imshow("rightMask", rightMask);
		// imshow("left", left);
		// imshow("right", right);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << "Time on export gait data: " << time.count() << std::endl;
		saver.save(lData, rData);
		lData.clear();
		rData.clear();
		if (waitKey(30) == 27) break;
	}
	return 0;
}