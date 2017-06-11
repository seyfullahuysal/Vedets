#include <iostream>
#include <opencv2/opencv.hpp>

#include "PixelBasedAdaptiveSegmenter.h"
#include "FrameDifferenceBGS.h"
#include "BlobTracking.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;
using namespace cv;

struct svm_model* training();
double test(float[], svm_model *);

struct svm_parameter param;     // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;

RNG rng(12345);

float Min_Al(Point2f giris[], int deger)
{
	float max = giris[0].y;
	for (int i = 0; i<deger; i++)
	{
		if (giris[i].y<max)
			max = giris[i].y;
	}
	return max;
}
float Min_Alx(Point2f giris[],int deger)
{
	float max = giris[0].x;
	for (int i = 0; i<deger; i++)
	{
		if (giris[i].x<max)
			max = giris[i].x;
	}
	return max;
}
Mat DikdorgenAl(vector<Point> contoursAs,Mat img_input)
{
	Mat sonuc = img_input;
	vector<cv::Point> noktalar;
	for (size_t j = 0; j < contoursAs.size(); j++) {
		cv::Point p = contoursAs[j];
		noktalar.push_back(p);
	}
	if (noktalar.size() > 0){
		cv::Rect brect = cv::boundingRect(cv::Mat(noktalar).reshape(2));
		return sonuc(brect);
	}
}
void CircleDetection(vector<Point> contoursAs, Mat sonuc,Mat &isle)
{
	Mat oImg, tImg,gecici;
	bool son = false;
	///////////////////////////////////
	
	vector<cv::Point> noktalar;
	for (size_t j = 0; j < contoursAs.size(); j++) {
		cv::Point p = contoursAs[j];
		noktalar.push_back(p);
	}
	if (noktalar.size() > 0){
		cv::Rect brect = cv::boundingRect(cv::Mat(noktalar).reshape(2));
		tImg = sonuc(brect);
		//////////////////////////////////
		cv::GaussianBlur(tImg, tImg, cv::Size(7, 7), 1, 1);

		cv::Canny(tImg, tImg, 10, 200);
		vector<Vec3f> circles;
		cv::HoughCircles(tImg, circles, CV_HOUGH_GRADIENT, 1, tImg.cols / 4);
		if (circles.size()!=0)
		{
			cout << "Cember Var" << endl;

		}
		else
		{
			vector<Point> contours_poly;
			Point2f center;
			float radius;
			approxPolyDP(Mat(contoursAs), contours_poly, 3, true);
			minEnclosingCircle((Mat)contours_poly, center, radius);
			cv::circle(isle, center, radius, Scalar(255, 0, 0), -1);
			
		}

	}
	
	

}
float Max_Contours(vector<Point> contours)
{
	float max = contours[0].y;
	for (int i = 0; i <contours.size(); i++)
	{
		if (contours[i].y>max)
			max = contours[i].y;
	}
	return max;
}

Mat Kontrol(Mat sonuc)
{
	Mat grayyy;
	if (sonuc.channels() == 3)
	{
		cvtColor(sonuc, grayyy, CV_RGB2GRAY);
	}
	else
	{
		grayyy = sonuc;
	}
	return grayyy;
}
Mat findConnectedComponents(Mat inputImage, int deger)
{
	Mat grayscale;
	if (inputImage.channels() == 3)
	{
		cvtColor(inputImage, grayscale, CV_BGR2GRAY);
	}
	else
	{
		grayscale = inputImage;
	}

	cv::threshold(grayscale, grayscale, 128, 255, CV_THRESH_BINARY);


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	findContours(grayscale, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point> > contours_polyg(contours.size());
	vector<Point2f>centerg(contours.size());
	vector<float>radiusg(contours.size());


	for (int i = 0; i < contours.size(); i++)
	{

		approxPolyDP(Mat(contours[i]), contours_polyg[i], 3, true);
		minEnclosingCircle((Mat)contours_polyg[i], centerg[i], radiusg[i]);
		if (radiusg[i] < deger)
		{
			contours[i].clear();
		}

	}

	Mat dst = Mat::zeros(inputImage.size(), CV_8UC1);
	if (!contours.empty() && !hierarchy.empty())
	{

		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar color(255, 255, 255);
			drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy, 0, cv::Point());
		}
	}


	return dst;
}
vector<vector<Point>> Kontur_Dondur(Mat inputImage)
{
	Mat grayscale;
	if (inputImage.channels() == 3)
	{
		cvtColor(inputImage, grayscale, CV_BGR2GRAY);
	}
	else
	{
		grayscale = inputImage;
	}

	cv::threshold(grayscale, grayscale, 128, 255, CV_THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	findContours(grayscale, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point> > contours_polyg(contours.size());
	vector<Point2f>centerg(contours.size());
	vector<float>radiusg(contours.size());


	for (int i = 0; i < contours.size(); i++)
	{

		approxPolyDP(Mat(contours[i]), contours_polyg[i], 3, true);
		minEnclosingCircle((Mat)contours_polyg[i], centerg[i], radiusg[i]);
		if (radiusg[i] < 15)
		{
			contours[i].clear();
		}

	}
	return contours;
}

int main(int argc, char **argv)
{
	
	/*FILE *dataset;
	if ((dataset = fopen("data.txt", "w+")) == NULL)
	{
		printf("data.txt Acilmadi!!!");
	}*/
		
	
	struct svm_model *model2 = training();


	int resize_factor = 25;
	CvCapture *capture = 0;
	capture = cvCaptureFromAVI("D:/Yazilim/Bitirme_Vedets/bitirme_dataset/dataset1.avi");

	if (!capture){
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}
	IplImage *frame_aux = cvQueryFrame(capture);
	IplImage *frame = cvCreateImage(cvSize((int)((frame_aux->width*resize_factor) / 100), (int)((frame_aux->height*resize_factor) / 100)), frame_aux->depth, frame_aux->nChannels);
	cvResize(frame_aux, frame);
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	IBGS *bgs;
	FrameDifferenceBGS *bgs2;
	bgs = new PixelBasedAdaptiveSegmenter;
	bgs2 = new FrameDifferenceBGS;
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	Mat img_mask, img_maskF, img_bkgmodel, img_input2,sonuc,grayyy;	

	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;

	while (key != 'q')
	{
		frame_aux = cvQueryFrame(capture);
		if (!frame_aux) break;
		cvResize(frame_aux, frame);
		Mat img_input(frame), son(frame);
		imshow("Giris", img_input);
		////////////////////////////////////////////////////
		img_input2 = img_input;
		Mat sonuc2;
		bgs->process(img_input, img_mask);
		bgs2->process(img_input2, img_maskF, img_bkgmodel);

		if (!img_maskF.empty())
		{

			sonuc = max(img_mask, img_maskF);
			sonuc = findConnectedComponents(sonuc, 15);
			cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4), cv::Point(-1, -1));
			cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2), cv::Point(-1, -1));
			cv::morphologyEx(sonuc, sonuc, cv::MORPH_OPEN, element, cv::Point(-1, -1), 4);
			cv::morphologyEx(sonuc, sonuc, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
			cv::morphologyEx(sonuc, sonuc, cv::MORPH_ERODE, element2, cv::Point(-1, -1), 1);
			cv::medianBlur(sonuc, sonuc, 5);
			cv::imshow("Maske", sonuc);
			////////////////////////////////////////////////////////////////////////////////////
			grayyy = Kontrol(sonuc);
			cv::threshold(grayyy, grayyy, 128, 255, CV_THRESH_BINARY);
			vector<vector<Point> > contours, contoursAs;
			vector<Vec4i> hierarchyAs;
			findContours(grayyy, contoursAs, hierarchyAs, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
			///////////////////////////////////////////////////////////////////////////////////
			vector<vector<Point> > contours_polyg(contoursAs.size());
			vector<Point2f>centerg(contoursAs.size());
			vector<float>radiusg(contoursAs.size());
			
			for (int i = 0; i < contoursAs.size(); i++)
			{
				approxPolyDP(Mat(contoursAs[i]), contours_polyg[i], 3, true);
				minEnclosingCircle((Mat)contours_polyg[i], centerg[i], radiusg[i]);
				if (radiusg[i] > 15 && contoursAs[i].size()>80)
				{
					contours.push_back(contoursAs[i]);
				}
			}
			if (contours.size() > 0)
			{
				vector<RotatedRect> minRect(contours.size());
				vector<RotatedRect> minEllipse(contours.size());
				for (int i = 0; i < contours.size(); i++)
				{
					minRect[i] = minAreaRect(Mat(contours[i]));
					if (contours[i].size() > 5)
					{
						minEllipse[i] = fitEllipse(Mat(contours[i]));
					}
				}

				Scalar color = Scalar(255, 0, 0);
				Scalar color2 = Scalar(0, 255, 0);

				float en = 0, boy = 0, imge_alan = 0, ellipse_major = 0, ellipse_minor = 0, tum = img_input.cols*img_input.rows;

				for (int i = 0; i < contours.size(); i++)
				{
					Point2f rect_points[4];
					minRect[i].points(rect_points);

					float kontrol_et = rect_points[0].y;
					

					if ((kontrol_et > img_input.rows - 27 &&  kontrol_et < img_input.rows - 5 ) && ((abs(rect_points[0].x - rect_points[3].x) == abs(rect_points[1].x - rect_points[2].x)) && (abs(rect_points[0].y - rect_points[1].y) == abs(rect_points[2].y - rect_points[3].y))))
					{
						en = minRect[i].size.width;
						boy = minRect[i].size.height;
						imge_alan = contourArea(contours[i]);

						if (minEllipse[i].size.width > minEllipse[i].size.height)
						{
							ellipse_major = minEllipse[i].size.width;
							ellipse_minor = minEllipse[i].size.height;
						}
						else
						{
							ellipse_major = minEllipse[i].size.height;
							ellipse_minor = minEllipse[i].size.width;
						}


						/*
						cout << i << "." << " bolgenin alan alan orani" << imge_alan/tum << endl;
						cout << i << "." << " bolgenin dikdortgen alan orani-->>" << " en oraný: "<<en/tum<<" boy: "<<boy/tum<<endl;
						cout <<i<< ". dikdorgen/imge_alan:" << ((en*boy) / imge_alan)<<endl;
						cout << i << " . Beyaz/Siyah: " << ((imge_alan) / ((en*boy) - imge_alan)) << endl;
						cout << i << ". Major alan orani" << ellipse_major/tum<<endl;
						cout << i << " . Minor alan orani" << ellipse_minor/tum<<endl;
						cout << "*********************************************************************" << endl;
						*/
						//////////////////////////////
						
						/*fprintf(dataset, "%f,", imge_alan / tum);
						fprintf(dataset, "%f,", en / tum);
						fprintf(dataset, "%f,", boy / tum);
						fprintf(dataset, "%f,", ((en*boy) / imge_alan));
						fprintf(dataset, "%f,", ellipse_minor / tum);
						fprintf(dataset, "%f,", ellipse_major / tum);
						fprintf(dataset, "%f\n", ((imge_alan) / ((en*boy) - imge_alan)));
						
						*/
						///////////////////////////////
						string yazdirr;
						float test_data[] = { imge_alan / tum, en / tum, boy / tum, ((en*boy) / imge_alan), ellipse_minor / tum, ellipse_major / tum, ((imge_alan) / ((en*boy) - imge_alan)) };
					
						
						if (test(test_data, model2) == 1)
						{
							yazdirr="kucuk";

						}
						else if (test(test_data, model2) == 2)
						{
							yazdirr = "buyuk";
						}
						else{
							yazdirr = "orta";
						}
						
						cout << "Aracin Sinifi: " << yazdirr << endl;
						
						putText(son, yazdirr, cv::Point(Min_Alx(rect_points, 4), Min_Al(rect_points,4) - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 1);
						
						
						for (int j = 0; j < 4; j++)
							cv::line(son, rect_points[j], rect_points[(j + 1) % 4], color, 2, 8);

					}
				}


			}
			imshow("Sýnýflama Sonuc", son);
			key = cvWaitKey(1);
		}
	}
	//fclose(dataset);
	
	delete bgs;

	cvDestroyAllWindows();
	cvReleaseCapture(&capture);

	return 0;
	
}
svm_model* training()
{

	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.5;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;


	//Problem definition-------------------------------------------------------------
	prob.l = 6754;
	int ozellik_sayisi = 7;
	//x values matrix of xor values

	FILE *dosya;
	if ((dosya = fopen("son_buyuk.arff", "r")) == NULL)
	{

		printf("dosya acilamadi 5 !!!");
		exit(0);

	}

	float matrix[6754][7];
	float y[6754];
	int i = 0;
	while (!feof(dosya))
	{
		fscanf(dosya, "%f %f %f %f %f %f %f %f\n", &matrix[i][0], &matrix[i][1], &matrix[i][2], &matrix[i][3], &matrix[i][4], &matrix[i][5], &matrix[i][6], &y[i]);
		i++;
	}

	//This part i have trouble understanding
	svm_node** x = Malloc(svm_node*, prob.l);

	//Trying to assign from matrix to svm_node training examples
	for (int row = 0; row <prob.l; row++){
		svm_node* x_space = Malloc(svm_node, ozellik_sayisi + 1);
		for (int col = 0; col < ozellik_sayisi; col++){
			x_space[col].index = col;
			x_space[col].value = matrix[row][col];
		}
		x_space[ozellik_sayisi].index = -1;      //Each row of properties should be terminated with a -1 according to the readme
		x[row] = x_space;
	}

	prob.x = x;

	//yvalues
	prob.y = Malloc(double, prob.l);


	for (int i = 0; i<prob.l; i++)
	{
		prob.y[i] = y[i];
	}

	//Train model---------------------------------------------------------------------
	model = svm_train(&prob, &param);
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	return model;

}




double test(float ozellik[], struct svm_model *model)
{
	//Test model----------------------------------------------------------------------
	svm_node* testnode = Malloc(svm_node, 7);
	testnode[0].index = 0;
	testnode[0].value = ozellik[0];
	testnode[1].index = 1;
	testnode[1].value = ozellik[1];
	testnode[2].index = 2;
	testnode[2].value = ozellik[2];
	testnode[3].index = 3;
	testnode[3].value = ozellik[3];
	testnode[4].index = 4;
	testnode[4].value = ozellik[4];
	testnode[5].index = 5;
	testnode[5].value = ozellik[5];
	testnode[6].index = 6;
	testnode[6].value = ozellik[6];
	testnode[7].index = -1;
	//This works correctly:
	double retval = svm_predict(model, testnode);
	return retval;

}