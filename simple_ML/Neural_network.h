#ifndef neural_network_h
#define neural_network_h
#include"Methods.h"

static double sum_theta_squared(std::vector<holder_M> data)
{
	double sum = 0;
	for (auto i : data)
	{
		sum += cv::sum((*i).mul(*i))[0];
	}
	return sum;
}

static double neural_error(const holder_M& inputb, const holder_M& outputb, const std::vector<holder_M>& theta, double lambda)
{
	cv::Mat logb1, logb2;
	cv::log(logistic_function(*inputb), logb1);
	cv::log(logistic_function(1 - *inputb), logb2);
	return -cv::sum((*outputb).t() * logb1 + (1 - *outputb).t() * logb2)[0] / (*inputb).rows + lambda / (2 * (*inputb).rows) * sum_theta_squared(theta);
}
static std::vector<double> structure = { };


struct neural_network :public methods 
{
public:
	neural_network() = default;
	neural_network(holder_M&, holder_M&, std::vector<double>, cv::Mat(*forward_function)(cv::Mat)= &logistic_function,
		double (*cost)(const holder_M&, const holder_M&, const std::vector<holder_M>&, double) = &neural_error);

	virtual cv::Mat calculate(holder_M&) override;
	virtual double return_error() override;
protected:
	virtual void data_normalization(holder_M&) override;
	virtual void regression(holder_M&, holder_M&) override;
	cv::Mat forward_regression(holder_M&);
	void backward_propogation(holder_M&);
	void selection();
	void generate_theta_bias();
	void separate_data(holder_M&, holder_M&);
	void structure_check(std::vector<double>);
	cv::Mat(*calculate_y)(cv::Mat);
	double (*cost_function)(const holder_M&, const holder_M&, const std::vector<holder_M>&, double);
	holder_M test_I, cross_v_I, train_I;
	holder_M test_O, cross_v_O, train_O;
	holder_M input, output;
	std::vector<double> neuron_structure;
	std::vector<double> mean;
	std::vector<holder_M> theta;
	std::vector<holder_M> predicted_layer;



};

#endif // !neural_network_h