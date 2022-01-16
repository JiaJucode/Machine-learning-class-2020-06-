#ifndef ML_H
#define ML_H
#include <vector>
#include "Simple_regression.h"
#include "Kernel_regression.h"
#include "Neural_network.h"

template <typename T> std::ostream& operator<<(std::ostream& a, std::vector<T> vec)
{
	for (int i = 0; i != vec.size(); i++)
	{
		a << " [ " << vec[i] << " ] \n";
	}
	return a;
}

static void no_change(holder_M& input)
{
	;
}

static std::vector<double> struc = { };

struct ml 
{
public:
	friend methods;
	ml() = default;
	//ml(std::string&&);
	ml(std::string&& input_file_directory, std::string&& output_file_directory, std::string&& file_type, void(* img_oper)(holder_M&) = &no_change, double variance_percentage_maintained = 1);
	ml(cv::Mat& intput,cv::Mat& output, double variance_percentage_maintained = 1);
	ml(std::string&& input_file_dir, std::string&& file_type_I, std::string&& output_file_dir, std::string&& file_type_O, void(* img_oper)(holder_M&) = &no_change, double variance_percentage_maintained = 1);
	ml(std::string&& input_txt_file, std::string&& output_txt_file, double variance_percentage_maintained = 1);
	double print_error(int user_method = 0);
	cv::Mat predict(cv::Mat& user_input, int user_method = 0, int output_type = 0, std::vector<double> structure = struc);
	cv::Mat predict(std::string&& input_file_dir, std::string&& input_file_type,int user_method = 0, int output_type = 0,  std::vector<double> structure = struc);

private:
	//void save();
	std::shared_ptr<methods> m;
	std::string file_T = ".txt";
	std::vector<std::string> input_file_names;
	std::vector<std::string> output_file_names;
	std::vector<std::string> test_input_files;
	holder_M test_input;
	holder_M input;
	holder_M output;
	int input_method = 0;
	double variance_maintained;
	cv::Mat eigenvector;
	cv::Size s; 
	bool fix = false;
	void (* change_input)(holder_M&);
	std::vector<double> neuron_structure;

	void data_compression(holder_M&);
	void check_null(int);
	void read_img(std::vector<std::string>& dir, holder_M& holder, bool);
	void read_txt(std::vector<std::string>& dir, holder_M& holder, bool);
	inline void empty_check(std::vector<std::string> V) const;
	void Ftype_check(std::vector<std::string> V) const;
	void check_variance() const;

};









#endif
