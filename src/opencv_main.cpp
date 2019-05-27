#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	if( argc!=2 )
	{
		cout << "input error" << endl;	
	}
	String modelFile = "../../model/model.pb";

	//initialize network
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelFile);
	if (net.empty())
		return -1;

	//prepare blob
	Mat img = imread(argv[1], IMREAD_GRAYSCALE);  //这里按灰度图读入是跟模型有关
	if (img.empty())
		return -2;

	resize(img, img, Size(28, 28));
	img = 255 - img;
	img.convertTo(img, CV_32F);
	img = img / 255.0f;

	Mat inputBlob = cv::dnn::blobFromImage(img);
	net.setInput(inputBlob);

	
	TickMeter tm;  //统计inference用时
	tm.start();

	//make forward pass
	Mat result = net.forward();
	tm.stop();
	Point maxLoc;
	cout << result << endl;
	minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	cout << "inference result: " << maxLoc.x << endl; 
	cout << "Time elapsed: " << tm.getTimeSec() << "s" << endl;
	
	return 1;
} //main


