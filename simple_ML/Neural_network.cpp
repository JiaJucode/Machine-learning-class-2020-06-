#include "Neural_network.h"
#include <numeric>


neural_network::neural_network(holder_M& I, holder_M& O, std::vector<double> network_structure, cv::Mat(*forward_function)(cv::Mat),
	double(*cost)(const holder_M&, const holder_M&, const std::vector<holder_M>&, double)):input(I), output(O), neuron_structure(network_structure)
{
	structure_check(neuron_structure);
	neuron_structure.insert(neuron_structure.begin(), (*input).cols);
	neuron_structure.push_back((*output).cols);
	calculate_y = forward_function;
	cost_function = cost;
	selection();
}

cv::Mat neural_network::calculate(holder_M& UI)
{
	for (int i = 0; i != (*UI).cols; i++) 
	{
		(*UI).col(i) = ((*UI).col(i) - mean[i]);
	}
	return forward_regression(UI);
}

double neural_network::return_error()
{
	forward_regression(test_I);
	return (*cost_function)(predicted_layer.back(), test_O, theta, lambda);
}

void neural_network::data_normalization(holder_M& data)
{
	for (int i = 0; i != (*data).cols; i++)
	{
		mean.push_back(cv::sum((*data).col(i))[0] / (*data).rows);
		(*data).col(i) -= mean[i];
	}
}

void neural_network::regression(holder_M& inputb, holder_M& outputb)
{
	generate_theta_bias();
	int loop_numb = 0;
	double pre_error;
	while (loop_numb < 20000) 
	{
		forward_regression(inputb);
		if (loop_numb == 0) 
		{
			pre_error = (*cost_function)(predicted_layer.back(), outputb, theta, lambda);
		}
		backward_propogation(outputb);
		if ((loop_numb) % check_freq == 0 && loop_numb != 0) {
			double error_b = (*cost_function)(predicted_layer.back(), outputb, theta, lambda);
			//std::cout << error_b << std::endl << *theta[0] << std::endl << *theta[1] << std::endl << (*predicted_layer[0]).row(0) << std::endl << (*predicted_layer[1]).row(0)
				//<< std::endl << (*predicted_layer[2]).row(0) << std::endl;
			if (error_b <= pre_error * 1.0000001) {
				if ((error_b / pre_error) >= loop_threshold || error_b == 0)
				{
					break;
				}
				else
				{
					alphal *= 1.2;
					pre_error = error_b;
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

cv::Mat neural_network::forward_regression(holder_M& data)
{
	if (!predicted_layer.empty()) {
		if (predicted_layer[0] != nullptr)
		{
			for (int i = 0; i != theta.size() + 1; i++)
			{
				(predicted_layer[i]).reset();
			}
		}
	}
	predicted_layer.resize(theta.size() + 1);	
	predicted_layer[0] = data;
	for (int i = 0; i != theta.size(); i++)
	{
		cv::Mat buffer(cv::Mat(1, (*data).rows, CV_64FC1, cv::Scalar(1)));
		buffer.push_back((*predicted_layer[i]).t());
		predicted_layer[i+1] = std::make_shared<cv::Mat>((*calculate_y)(buffer.t() * *theta[i]));

	}
	return *predicted_layer[predicted_layer.size() - 1];
}

void neural_network::backward_propogation(holder_M& O)
{
	cv::Mat previous_delta;
	for (int i = theta.size() - 1; i != -1; i--)
	{
		if (i == theta.size() - 1) {
			previous_delta = (*predicted_layer[i + 1] - *O).mul((*calculate_y)(*predicted_layer[i + 1]).mul(1 - (*calculate_y)(*predicted_layer[i + 1])));
		}
		else 
		{
			cv::Mat buf = previous_delta * (*theta[i + 1]).t();
			previous_delta = (buf.colRange(1,buf.cols)).mul((*calculate_y)(*predicted_layer[i + 1]).mul(1 - (*calculate_y)(*predicted_layer[i + 1])));
		}
		cv::Mat predicted_l2(1, (*predicted_layer[i]).rows, CV_64FC1, cv::Scalar(1));
		predicted_l2.push_back((*predicted_layer[i]).t());
		*theta[i] -= alphal / (*predicted_layer[0]).rows * (predicted_l2 * previous_delta + lambda * *theta[i]);

	}	
}

void neural_network::selection()
{
	double starting_time = cv::getTickCount();
	data_normalization(input);
	double alpha_buffer = alphal;
	int loop_times = 6;
	int loop_numb = 0;
	double train_error, cross_error;
	while (loop_numb != loop_times) 
	{
		separate_data(input, output);
		regression(train_I, train_O);
		loop_numb++;
		train_error = (*cost_function)(predicted_layer.back(), train_O, theta, lambda);
		forward_regression(cross_v_I);
		cross_error = (*cost_function)(predicted_layer.back(), cross_v_O, theta, lambda);
		double threshold = error_threshold + lambda / (2 * (*train_I).rows) * sum_theta_squared(theta);
		if (cross_error > threshold && train_error > threshold)
		{

			std::cout << "underfitting detected, recommand changing the neuron structure or adding more training data. loop number: " << loop_numb << std::endl;
			break;
			//throw(std::runtime_error("underfit"));
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
	}
	alphal = alpha_buffer;
	std::cout << std::endl << "Neural network training report: " << std::endl << "Error of prediction on training set: " << train_error
		<< std::endl << "Error of prediction on cross validation set: " << cross_error
		<< std::endl << "number of times the training was performed: " << loop_numb
		<< std::endl << "lambda value: " << lambda 
		<< std::endl << "overall training time: " << (cv::getTickCount()-starting_time)/cv::getTickFrequency() << std::endl;
}

void neural_network::generate_theta_bias()
{
	if (!theta.empty() && theta[0] != nullptr)
	{
		for (auto x : theta)
		{
			x.reset();
		}
	}
	theta.resize(neuron_structure.size() - 1);
	for (int i = 0; i != neuron_structure.size() - 1; i++)
	{
		holder_M buffer = std::make_shared<cv::Mat>(cv::Mat(neuron_structure[i] + 1, neuron_structure[i + 1], CV_64FC1));
		cv::randu(*buffer, cv::Scalar(-10), cv::Scalar(10));
		theta[i] = buffer;
	}

}

void neural_network::separate_data(holder_M& I, holder_M& O)
{
	if ((*I).rows != (*O).rows)
	{
		std::cout << "input and output have different numbers of data" << std::endl << "input: " << (*I).rows << std::endl
			<< "output: " << (*O).rows << std::endl;
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

void neural_network::structure_check(std::vector<double> buffer)
{
	for (auto i : buffer) 
	{
		if (i == 0) 
		{
			std::cout << "user defined network strcture cannot have a layer with 0 neurons" << std::endl;
			throw(std::runtime_error("incorrect structure"));
		}
	}
}


//include theta to cost function