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
// 压力数据类，用于存放压力数据
class PressureData
{
public:
	double Mx, My, force, Fx, Fy; 
	// contours是脚的轮廓
	vector<vector<cv::Point>> contours;
	PressureData();
	PressureData(double Mx, double My, double force);
	// 运算符重载，使得我们的类的对象支持如下操作：data1 = data2
	PressureData& operator=(const PressureData& pData)
	{
		// 处理自我赋值，如 data = data的情形
		if (this == &pData) return *this;
		Mx = pData.Mx;
		My = pData.My;
		force = pData.force;
		Fx = pData.Fx;
		Fy = pData.Fy;
		contours = pData.contours;
		// 处理链式赋值
		return *this;
	}
	// 用友元函数交换两个对象的数值，注意使用了引用参数
	friend void swap(PressureData& p1, PressureData& p2)
	{
		swap(p1.Mx, p2.Mx);
		swap(p1.My, p2.My);
		swap(p1.force, p2.force);
		swap(p1.Fx, p2.Fx);
		swap(p1.Fy, p2.Fy);
		swap(p1.contours, p2.contours);
	}
	// 输出数据
	void printData()
	{
		printf("Mx: %f\tMy: %f\t Force: %f\tFx: %f\tFy: %f\n", Mx, My, force, Fx, Fy);
	}
	// 用于清空数据
	void clear()
	{
		Mx = My = force = Fx = Fy = -1;
		contours.clear();
	}
};
PressureData::PressureData()
{
	// -1表示无意义、不存在
	Mx = My = force = Fx = Fy = -1;
	
}
PressureData::PressureData(double _Mx, double _My, double _force)
{
	Mx = _Mx;
	My = _My;
	force = _force;
	// 压力中心的位置即力矩处以力的大小
	Fx = Mx / _force;
	Fy = My / _force;
}

// 这个类帮助我们保存数据到文件，选用csv文件，它可以很方便地用Excel打开为表格格式
// 而且csv又是一个文本文件，足够简单
class DataSaver
{
public:
	ofstream file;
	// 构造函数：打开文件并输出表格的头
	DataSaver()
	{
		file.open("data.csv");
		file<<"Left force,Left X,Left Y,Right force,Right X,Right Y\n";
	}
	// 析构函数关闭文件，无需手动关闭
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
	VideoWriter writer("result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, Size(56, 128));
	Mat frame;
	int delay = 1000/30;
	namedWindow("Frame", cv::WINDOW_NORMAL);
	// namedWindow("Binary", cv::WINDOW_NORMAL);
	// namedWindow("leftMask", cv::WINDOW_NORMAL);
	// namedWindow("rightMask", cv::WINDOW_NORMAL);
	// namedWindow("left", cv::WINDOW_NORMAL);
	// namedWindow("right", cv::WINDOW_NORMAL);
	
	// 创建一个结构，用于之后的膨胀操作
	Mat dilateEle = getStructuringElement(cv::MORPH_DILATE, cv::Size(5, 19));
	// 分别存放左、右教的数据
	PressureData lData;
	PressureData rData;
	while (true)
	{
		auto start = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if (frame.empty()) break;
		Mat grey;
		// 如果你想让图像看上去亮一些，取消下一行的注释
		// frame *= 10;
		// 下面这一行在原图像的头顶和左侧各添加一排数据，你也可以不添加
		// copyMakeBorder(frame, frame, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
		if (frame.channels() == 3) cvtColor(frame, grey, COLOR_BGR2GRAY);
		Mat binary;
		threshold(grey, binary, 0, 255, THRESH_BINARY);
		// 请思考一下膨胀的作用
		dilate(binary, binary, dilateEle, Point(-1,-1), 1);
		vector<vector<Point2i>> contours, leftContour, rightContour;
		findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		assert (contours.size()<3); // serious error, 之后加入报错机制

		// 没有轮廓，说明压力图中无数据，可以读入下一图片了
		if (contours.size()<1)
		{
			cout << "No foot detected!" << endl;
			writer << frame;
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
			// 如只有一个轮廓，说明只检测到一只脚
			Rect r = boundingRect(Mat(contours[0]));
			Point center = (r.br()+r.tl())/2; // 获取外接矩形的中心
			if (frame.cols/2 < center.x)
			{
				// 如果中心在图右侧，有可能是左脚
				// 画轮廓，其中-1参数表示把轮廓里面画满，即将脚型填充为白色
				drawContours(leftMask, contours, -1, Scalar(255), -1); 
				// 利用掩模操作，将轮廓内的数据复制到left，而其他部分不复制
				grey.copyTo(left, leftMask);
				// 现在left图像内，只有单只脚
				// 利用矩这个类，获取力矩
				Moments lM = moments(left);
				// 注意这个构造函数可以由力矩和力求得力臂的大小（即压力中心的位置）
				lData = PressureData(lM.m10, lM.m01, sum(left)[0]);
				lData.contours = contours;
				// 再检查一次有无错误
				if (lData.Fx < frame.cols/2) swap(lData, rData);
			}
			else
			{
				// 如果中心在图左侧，有可能是右脚
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
		else
		{
			cerr<<"More than 2 feet were detected !"<<endl;
		}
		
		// 有则画出框来
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
		writer << frame;
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
		if (waitKey(delay) == 27) break;
	}
	return 0;
}