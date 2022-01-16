#ifndef Kernel_regression_h
#define Kernel_regression_h
#include"Methods.h"


struct kernel_regression : public methods 
{
public:
	kernel_regression() = default;
	kernel_regression(holder_M&, holder_M&, void(*function_ptr)(holder_M&, holder_M&, holder_M&, double&, double&) = &regression_function,
		double(*cost_ptr)(const holder_M&, const holder_M&, const holder_M&, double) = &calculate_error, cv::Mat(*cal_y)(cv::Mat) = &logistic_function);

	virtual cv::Mat calculate(holder_M&) override;
	virtual double return_error() override;
private:
	virtual void data_normalization(holder_M&) override;
	virtual void regression(holder_M&, holder_M&) override;
	void separate_data(holder_M&, holder_M&);
	void selection();
	void edit_data(holder_M&, holder_M&);
	void (*forward_reg)(holder_M&, holder_M&, holder_M&, double&, double&);
	double (*cost)(const holder_M&, const holder_M&, const holder_M&, double);
	cv::Mat(*y_function)(cv::Mat);
	holder_M input;
	holder_M output;
	holder_M theta;
	holder_M test_I, cross_v_I, train_I;
	holder_M test_O, cross_v_O, train_O;
	holder_M kernel_input;
	double kernel_variance = 1;
	std::vector<double> mean;
	std::vector<double> variance;
};

#endif // !Kernel_regression_h
