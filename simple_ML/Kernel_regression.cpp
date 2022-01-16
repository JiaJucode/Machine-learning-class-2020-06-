#include"Kernel_regression.h"

kernel_regression::kernel_regression(holder_M& I, holder_M& O, void(*function_ptr)(holder_M&, holder_M&, holder_M&, double&, double&),
	double(*cost_ptr)(const holder_M&, const holder_M&, const holder_M&, double), cv::Mat(*cal_y)(cv::Mat)) :input(I), output(O)
{
	y_function = cal_y;
	forward_reg = function_ptr;
	cost = cost_ptr;
	selection();
}

cv::Mat kernel_regression::calculate(holder_M& UI)
{
	holder_M buffer;
	for (int i = 0; i != (*UI).cols; i++)
	{
		(*UI).col(i) = ((*UI).col(i) - mean[i]) / variance[i];
	}
	edit_data(UI, buffer);
	return (*y_function)(*buffer * *theta);
}

double kernel_regression::return_error()
{
	return (*cost)(test_I, test_O, theta, lambda);
}

void kernel_regression::data_normalization(holder_M& data)
{
	for (int i = 0; i != (*data).cols; i++)
	{
		mean.push_back(cv::sum((*data).col(i))[0] / (*data).rows);
		(*data).col(i) -= mean[i];
		cv::Mat buffer;
		cv::pow((*data).col(i), 2, buffer);
		variance.push_back(sum(buffer)[0] / (*data).rows);
		if (variance[i] == 0)
		{
			std::cout << "non-changing feature in colume: " << i << std::endl << (*data);
			throw(std::runtime_error("unnecessary data"));
		}
		(*data).col(i) /= variance[i];
	}
}

void kernel_regression::regression(holder_M& inputb, holder_M& outputb)
{
	cv::randu(*theta, cv::Scalar(-1), cv::Scalar(1));
	int loop_numb = 0;
	double pre_error = (*cost)(inputb, outputb, theta, lambda);
	while (loop_numb < 20000) {
		double buffer_c = alphal / (*inputb).rows;
		//std::cout << *theta << std::endl << std::endl;
		(*forward_reg)(inputb, outputb, theta, buffer_c, lambda);
		//std::cout << *theta << std::endl << std::endl;
		//system("pause");
		if (loop_numb % check_freq == 0 && loop_numb != 0) {
			double error_b = (*cost)(inputb, outputb, theta, lambda);
			//std::cout << pre_error << "  " << error_b << " " << alphal << std::endl;
			if (error_b <= pre_error * 1.0000001) {
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

void kernel_regression::separate_data(holder_M& I, holder_M& O)
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
		if (i < ceil(test_p * (*I).rows))
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

void kernel_regression::selection()
{
	double starting_time = cv::getTickCount();
	int loop_numb = 0;
	int loop_times = 6;
	data_normalization(input);
	double alpha_buffer = alphal;
	cv::Mat smallest_theta((*input).rows, (*input).rows, CV_64FC1);
	double smallest_v = 0;
	double smallest_error = 99999999;
	double train_error, cross_error;
	while (loop_numb < loop_times + 1) {
		edit_data(input, kernel_input);
		double threshold1, threshold2;
		if (theta == nullptr) {
			theta = std::make_shared<cv::Mat>(cv::Mat((*kernel_input).cols, (*output).cols, CV_64FC1));
		}
		else
		{
			theta.reset(new cv::Mat((*kernel_input).cols, (*output).cols, CV_64FC1));
		}
		separate_data(kernel_input, output);
		regression(train_I, train_O);
		cross_error = (*cost)(cross_v_I, cross_v_O, theta, lambda);
		train_error = (*cost)(train_I, train_O, theta, lambda);
		//system("pause");
		threshold1 = error_threshold + lambda / (2 * (*cross_v_I).rows) * sum((*theta).mul(*theta))[0];
		threshold2 = error_threshold + lambda / (2 * (*train_I).rows) * sum((*theta).mul(*theta))[0];
		if (cross_error < smallest_error) 
		{
			smallest_theta = *theta;
			smallest_v = kernel_variance;
		}
		if (cross_error > threshold1 && train_error > threshold2)
		{
			if (lambda - 0.1 <= 0) {
				kernel_variance *= 1.1;
			}
			else
				lambda -= 0.1;
		}
		else
		{
			if (cross_error > threshold1 && train_error < threshold2)
			{
				lambda += 0.1;
			}
			else
			{
				break;
			}
		}
		alphal = alpha_buffer;
		loop_numb++;
	}
	*theta = smallest_theta;
	kernel_variance = smallest_v;
	edit_data(input, kernel_input);
	//std::cout << loop_numb << " " << lambda << " " << kernel_variance << std::endl;
	std::cout << "Kernel regression training report: " << std::endl << "Error of prediction on training set: " << train_error
		<< std::endl << "Error of prediction on cross validation set: " << cross_error
		<< std::endl << "number of times the training was performed: " << loop_numb
		<< std::endl << "lambda value: " << lambda
		<< std::endl << "kernel variance: " << kernel_variance
		<< std::endl << "overall training time: " << (cv::getTickCount() - starting_time) / cv::getTickFrequency() << std::endl;
}

double calculate_kernel(cv::Mat current, double var) 
{
	cv::Mat buffer;
	cv::pow(current, 2, buffer);
	return std::exp(-std::sqrt(cv::sum(buffer)[0])/(2*std::pow(var,2)));
}

void kernel_regression::edit_data(holder_M& data, holder_M& kernel_input)
{
	if (kernel_input != nullptr) {
		kernel_input.reset();
	}
	kernel_input = std::make_shared<cv::Mat>(cv::Mat((*data).rows, (*input).rows, CV_64FC1));
	for (int i = 0; i != (*data).rows; i++) 
	{
		for (int j = 0; j != (*input).rows; j++) 
		{
			(*kernel_input).at<double>(i, j) = calculate_kernel((*data).row(i)-(*input).row(j), kernel_variance);
		}
	}

}
