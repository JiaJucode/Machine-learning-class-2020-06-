#include "Simple_regression.h"

simple_regression::simple_regression(holder_M& I, holder_M& O, void(*function_ptr)(holder_M&, holder_M&, holder_M&, double&, double&), 
	double(*cost_ptr)(const holder_M&, const holder_M&, const holder_M&, double), cv::Mat(*cal_y)(cv::Mat), int dim) :dimension(dim), input(I), output(O)
{
	original = std::make_shared<cv::Mat>(cv::Mat());
	*original = *I;
	y_function = cal_y;
	forward_reg = function_ptr;
	cost = cost_ptr;
	selection();
}

void simple_regression::selection()
{
	double starting_time = cv::getTickCount();
	int loop_numb = 0;
	int loop_times = 6;
	double alpha_buffer = alphal;
	double train_error, cross_error;
	while (loop_numb < loop_times + 1) {
		if (theta == nullptr) {
			theta = std::make_shared<cv::Mat>(cv::Mat((*original).cols * dimension + 1, (*output).cols, CV_64FC1));
		}
		else 
		{
			mean.clear();
			variance.clear();
			theta.reset(new cv::Mat((*original).cols * dimension + 1, (*output).cols, CV_64FC1));
		}
		double threshold;
		edit_data(input);
		data_normalization(input);
		separate_data(input, output);
		regression(train_I, train_O);
		cross_error = (*cost)(cross_v_I, cross_v_O, theta, lambda);
		train_error = (*cost)(train_I, train_O, theta, lambda);
		threshold = error_threshold + lambda / (2 * (*original).rows) * sum((*theta).mul(*theta))[0];
		if (cross_error > threshold && train_error > threshold) 
		{
			if (loop_numb != loop_times) {
				dimension += 1;
			}
		}
		else 
		{
			//std::cout << cross_error << " " << threshold << " " << train_error;
			if (cross_error > threshold && train_error < threshold) 
			{
				lambda += 0.1;
			}
			else 
			{
				break;
			}
		}
		loop_numb++;
		alphal = alpha_buffer;
		*input = *original;
	}
	std::cout << "Simple regression training report: " << std::endl << "Error of prediction on training set: " << train_error
		<< std::endl << "Error of prediction on cross validation set: " << cross_error
		<< std::endl << "number of times the training was performed: " << loop_numb
		<< std::endl << "lambda value: " << lambda
		<< std::endl << "extra features added: " << dimension - 1
		<< std::endl << "lambda value: " << lambda
		<< std::endl << "overall training time: " << (cv::getTickCount() - starting_time) / cv::getTickFrequency() << std::endl;
}

void simple_regression::data_normalization(holder_M&data)
{
	for (int i = 1; i != (*data).cols; i++) 
	{
		mean.push_back(cv::sum((*data).col(i))[0]/(*data).rows);
		(*data).col(i) -= mean[i - 1];
		cv::Mat buffer;
		cv::pow((*data).col(i), 2, buffer);
		variance.push_back(sum(buffer)[0] / (*data).rows);
		if (variance[i-1] == 0) 
		{
			std::cout << "non-changing feature in colume: " << i << std::endl << (*data);;
			throw(std::runtime_error("unnecessary data"));
		}
		(*data).col(i) /= variance[i-1];
	}
}

void simple_regression::edit_data(holder_M& data)
{
	cv::transpose(*data, *data);
	cv::Mat one(1, (*data).cols, CV_64FC1, cv::Scalar(1));
	one.push_back(*data);
	(*data) = one;
	for (int i = 1; i != dimension; i++) {
		cv::Mat buffer;
		cv::pow(((*data)(cv::Range(1, (*original).cols + 1), cv::Range(0, (*data).cols))), i + 1, buffer);
		(*data).push_back(buffer);
	}
	cv::transpose(*data, *data);
}

void simple_regression::regression(holder_M& inputb, holder_M& outputb)
{
	cv::randu(*theta, cv::Scalar(-1), cv::Scalar(1));
	int loop_numb = 0;
	double pre_error = (*cost)(inputb, outputb, theta, lambda);
	while (loop_numb<20000) {
		double buffer_c = alphal / (*inputb).rows;
		(*forward_reg)(inputb, outputb, theta, buffer_c, lambda);
		if (loop_numb % check_freq == 0 && loop_numb != 0) {
			double error_b = (*cost)(inputb, outputb, theta, lambda);
			if (error_b <= pre_error*1.0000001) {
				if ((error_b / pre_error) >= loop_threshold || error_b == 0)
				{
					break;
				}
				else
				{
					alphal *= 1.2;
					pre_error = (*cost)(inputb, outputb, theta, lambda);
				}
			}
			else 
			{
				alphal /= 1.2;
			}
		}
		loop_numb++;
	}
}

cv::Mat simple_regression::calculate(holder_M& UI)
{
	cv::Mat return_value((*UI).rows,(*UI).cols,CV_64FC1,cv::Scalar(0));
	edit_data(UI);
	for (int i = 1; i != (*UI).cols; i++) 
	{
		(*UI).col(i) = ((*UI).col(i) - mean[i - 1]) / variance[i - 1];
	}
	return_value = (*y_function)(*UI * *theta);
	return return_value;
}

double simple_regression::return_error()
{
	return (*cost)(test_I, test_O, theta, lambda);
}

void simple_regression::separate_data(holder_M& I, holder_M& O)
{
	if ((*I).rows != (*O).rows)
	{
		throw(std::runtime_error("arguments for mix_data should have the same size"));
	}
	for (int i = 0; i != (*I).rows; i++) 
	{
		int r_index = rand() % int((*I).rows);
		if (r_index == i) 
		{
			if (r_index == (*I).rows - 1) 
			{
				r_index -= 1;
			}
			else 
			{
				r_index += 1;
			}
		}
		cv::Mat b1; (*I).row(i).copyTo(b1);
		cv::Mat b2; ((*I).row(r_index)).copyTo(b2);
		b2.copyTo((*I).row(i)); b1.copyTo((*I).row(r_index));
		((*O).row(i)).copyTo(b1);
		((*O).row(r_index)).copyTo(b2);
		b2.copyTo((*O).row(i)); b1.copyTo((*O).row(r_index));
	}
	test_I.reset(new cv::Mat());	test_O.reset(new cv::Mat());
	cross_v_I.reset(new cv::Mat());	cross_v_O.reset(new cv::Mat());
	train_I.reset(new cv::Mat());	train_O.reset(new cv::Mat());
	for (int i = 0; i != (*I).rows; i++)
	{
		if (i < ceil(test_p*(*I).rows))
		{
			(*test_I).push_back((*I).row(i));
			(*test_O).push_back((*O).row(i));
		}
		else 
		{
			if (i < ceil(test_p * (*I).rows) + ceil(cross_p * (*I).rows))
			{
				(*cross_v_I).push_back((*I).row(i));
				(*cross_v_O).push_back((*O).row(i));
			}
			else 
			{
				(*train_I).push_back((*I).row(i));
				(*train_O).push_back((*O).row(i));
			}
		}
	}
}
