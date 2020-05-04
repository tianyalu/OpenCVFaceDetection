// OpenCVFaceDetection.cpp: 定义应用程序的入口点。
//

#include "OpenCVFaceDetection.h"

void calcLBP(Mat, Mat&);

CascadeClassifier face_cascade_classifier;
cv::Ptr<DetectionBasedTracker> tracker;


int main()
{
	//D:/CommonDev/opencv/opencv4.1.2/opencv\build/etc/haarcascades/haarcascade_frontalface_alt.xml

	//Mat ones = Mat::ones(2, 2, CV_8UC3);
	//Mat ones = Mat::eye(3, 3, CV_8UC3);;
	//cout << ones << endl;

#ifdef DETECT
	if (!face_cascade_classifier.load("D:/CommonDev/opencv/opencv4.1.2/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml")) {
		cout << "级联分类器加载失败！" << endl;
		return -1;
	}
#elif DYNAMIC_DETECT
	//主检测适配器
	cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
		makePtr<CascadeClassifier>("D:/CommonDev/opencv/opencv4.1.2/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml"));
	//跟踪检测适配器
	cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
		makePtr<CascadeClassifier>("D:/CommonDev/opencv/opencv4.1.2/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml"));
	//创建跟踪器
	DetectionBasedTracker::Parameters DetectorParams;
	tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
	//启动跟踪器
	tracker->run();
#endif // DETECT

	

	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened()) {
		cout << "摄像头打开失败！" << endl;
		return -1;
	}

	Mat frame; //摄像头的图像
	Mat gray0, gray;

	while (true) {
		capture >> frame;
		if (frame.empty()) {
			cout << "摄像头采集图像失败！" << endl;
			return -1;
		}
		imshow("摄像头", frame); //彩色图
		//预处理
		cvtColor(frame, gray0, COLOR_RGB2GRAY); //转成灰度图， opencv中：bgr
		imshow("黑白相机", gray0);
		//equalizeHist(gray0, gray); //直方均衡化，增强对比度
		vector<Rect> faces;

#ifdef LBP_CALCULATE
		Mat lbp = Mat(gray0.rows - 2, gray0.cols - 2, CV_8UC1);
		calcLBP(gray0, lbp);
		imshow("lbp图", lbp);
#else


#ifdef DETECT
		face_cascade_classifier.detectMultiScale(gray, faces);
#elif DYNAMIC_DETECT
		//人脸动态检测处理
		tracker->process(gray);
		tracker->getObjects(faces);
#endif // DETECT
		int i = 0;
		for each (Rect face in faces) {  //VS特有的语法
			rectangle(frame, face, Scalar(0, 0, 255));

#ifdef COLECT_SAMPLES //采集样本
			Mat sample;
			frame(face).copyTo(sample);  //抠图（在彩图frame 上抠图）
			resize(sample, sample, Size(24, 24));  //归一化样本大小
			cvtColor(sample, sample, COLOR_BGR2GRAY); //灰度化处理
			char p[100];
			sprintf(p, "C:/Users/Administrator/Desktop/opencv/train/face/pos/%d.jpg", i++);
			imwrite(p, sample);  //保存样本图像，与imread() 读对应
#endif // COLECT_SAMPLES //采集样本

		}
		imshow("人脸检测", frame);

#endif //LBP_CALCULATE

		//等待30ms
		if (waitKey(30) == 27) { //ESC键
			break;
		}
	}

	return 0;
}

// 计算LBP图谱算法
// src灰度图
// Mat(高，宽)
void calcLBP(Mat src, Mat& dst) {
	for (int i = 1; i < src.rows - 1; i++) //高
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar lbp = 0;
			//取（i,j）位置的像素
			uchar center = src.at<uchar>(i, j);
			//左上角 顺时针
			if (src.at<uchar>(i - 1, j - 1) > center) {
				lbp += 1 << 7;
			}
			//正上方
			if (src.at<uchar>(i - 1, j) > center) {
				lbp += 1 << 6;
			}
			//右上方
			if (src.at<uchar>(i - 1, j + 1) > center) {
				lbp += 1 << 5;
			}
			//正右方
			if (src.at<uchar>(i, j + 1) > center) {
				lbp += 1 << 4;
			}
			//右下角
			if (src.at<uchar>(i + 1, j + 1) > center) {
				lbp += 1 << 3;
			}
			//正下方
			if (src.at<uchar>(i + 1, j) > center) {
				lbp += 1 << 2;
			}
			//左下方
			if (src.at<uchar>(i + 1, j - 1) > center) {
				lbp += 1 << 1;
			}
			//左
			if (src.at<uchar>(i, j - 1) > center) {
				lbp += 1 << 0;
			}

			dst.at<uchar>(i - 1, j - 1) = lbp;
		}
	}
}