#ifndef METHODS_H
#define METHODS_H
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <memory>

typedef std::shared_ptr<cv::Mat> holder_M;

static cv::Mat logistic_function(cv::Mat input)
{
	cv::Mat output(input.rows, input.cols, CV_64FC1);
	for (int i = 0; i != input.cols; i++) {
		for (int j = 0; j != input.rows; j++)
		{
			output.at<double>(j, i) = 1 / (1 + exp(-input.at<double>(j, i)));
		}
	}
	return output;
}

static void regression_function(holder_M& inputb, holder_M& outputb, holder_M& theta, double& multi, double& lambda)
{
	//std::cout << multi * ((*inputb).t() * (logistic_function(*inputb * *theta) - *outputb ) + lambda / *theta) << std::endl;
	(*theta) -= multi*((*inputb).t() * (logistic_function(*inputb * *theta) - *outputb) + lambda * *theta);
}

static double calculate_error(const holder_M& inputb, const holder_M& outputb, const holder_M& theta, double lambda)
{
	cv::Mat logb1, logb2;
	cv::log(logistic_function((*inputb) * *theta), logb1);
	cv::log(logistic_function((1 - *inputb) * *theta), logb2);
	return -cv::sum((*outputb).t() * logb1 + (1 - *outputb).t() * logb2)[0] / (*inputb).rows + lambda / (2 * (*inputb).rows) * sum((*theta).mul(*theta))[0];
}

struct ml;

struct methods 
{
public:


	virtual cv::Mat calculate(holder_M&) = 0;
	virtual ~methods() = default;
	virtual double return_error() = 0;
protected:
	virtual void data_normalization(holder_M&) = 0;
	virtual void regression(holder_M&, holder_M&) = 0;
	double lambda = 0;//underfit, overfit
	double alphal = 0.1;//learning rate
	const double error_threshold = 0.4;
	const double loop_threshold = 0.999;// the smaller the value, the lesser number of loop that the program will do and it will predict less accurate on the train data set 
	const int check_freq = 1000;// frequncy at which the loop threshold check is performed
	const double test_p = 0.2, cross_p = test_p, train_p = 1 - test_p - cross_p;
};

#endif // !METHODS_H
