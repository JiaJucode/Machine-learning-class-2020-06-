#include "ML.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <fstream>

/*
ml::ml(std::string&&)
{

}
*/
ml::ml(std::string&& input_file_directory, std::string&& output_file_directory, std::string&& file_type, void (* img_oper)(holder_M&), double variance_percentage_maintained)
	:file_T(file_type), change_input(img_oper), variance_maintained(variance_percentage_maintained)
{
	cv::glob(input_file_directory, input_file_names, false);
	cv::glob(output_file_directory, output_file_names, false);
	empty_check(input_file_names);	empty_check(output_file_names);
	Ftype_check(input_file_names);	Ftype_check(output_file_names);
	check_variance();
	read_img(input_file_names, input, 0);
	read_img(output_file_names, output, 1);
}

ml::ml(cv::Mat& inp,cv::Mat& out, double variance_percentage_maintained):variance_maintained(variance_percentage_maintained)
{
	check_variance();
	if (inp.rows == out.rows) {
		input = std::make_shared<cv::Mat>(inp);
		output = std::make_shared<cv::Mat>(out);
	}
	else 
	{
		std::cout << "inconsistant numbers of input and output\n input: " << inp.cols << " output: " << out.cols;
		throw(std::runtime_error("inconsistant input"));
	}
}

ml::ml(std::string&& input_file_dir, std::string&& file_type_I, std::string&& output_file_dir, std::string&& file_type_O, void (* img_oper)(holder_M&), double variance_percentage_maintained)
	:change_input(img_oper), variance_maintained(variance_percentage_maintained)
{
	check_variance();
	if (file_type_O == ".txt")
	{
		output_file_names.push_back(output_file_dir);
		file_T = file_type_O;
		Ftype_check(output_file_names);
		read_txt(output_file_names, output, 1);
	}
	else
	{
		cv::glob(output_file_dir, output_file_names, false);
		file_T = file_type_O;
		empty_check(output_file_names);	Ftype_check(output_file_names);
		read_img(output_file_names, output, 1);
	}
	if (file_type_I == ".txt")
	{
		input_file_names.push_back(input_file_dir);
		file_T = file_type_I;
		Ftype_check(input_file_names);
		read_txt(input_file_names, input, 0);
	}
	else
	{
		cv::glob(input_file_dir, input_file_names, false);
		file_T = file_type_I;
		empty_check(input_file_names);	Ftype_check(input_file_names);
		read_img(input_file_names, input, 0);
	}
}

ml::ml(std::string&& input_txt_file, std::string&& output_txt_file, double variance_percentage_maintained): variance_maintained(variance_percentage_maintained)
{
	input_file_names.push_back(input_txt_file);
	output_file_names.push_back(output_txt_file);
	Ftype_check(input_file_names);	Ftype_check(output_file_names);
	check_variance();
	read_txt(input_file_names, input, 0);
	read_txt(output_file_names, output, 1);
}

double ml::print_error(int user_method)
{
	check_null(user_method);
	double return_buffer = (*m).return_error();
	std::cout << "average error of prediction: " << return_buffer << std::endl;
	return return_buffer;
}

void ml::data_compression(holder_M& a)
{
	cv::PCA pca(*a, cv::Mat(), cv::PCA::DATA_AS_ROW);
	eigenvector = pca.eigenvectors;
	if (variance_maintained != 1) 
	{
		double v_percentage = 1;
		int counter = 1;
		double base = cv::sum(pca.eigenvalues)[0];
		while (v_percentage >= variance_maintained)
		{
			v_percentage = (base - cv::sum((pca.eigenvalues).rowRange((pca.eigenvalues).rows - counter, (pca.eigenvalues).rows))[0]) / base;
			counter++;
		}
		counter--;
		eigenvector = (pca.eigenvectors).rowRange(0, (pca.eigenvectors).rows - counter);
	}
	*a = *a * (eigenvector).t();
	//std::cout << (*a).cols << " " << (*a).rows << std::endl;
	//std::cout << (*a).cols << " " << (*a).rows;
}

void ml::check_null(int user_method)
{
	if (m == nullptr || input_method != user_method) {
		switch (user_method)
		{
		case 0:
		{
			m = std::shared_ptr<methods>(new simple_regression(input, output));
			break;
		}
		case 1:
		{
			m = std::shared_ptr<methods>(new kernel_regression(input, output));
			break;
		}
		case 2:
		{
			m = std::shared_ptr<methods>(new neural_network(input, output, neuron_structure));
			break;
		}
		default:
		{
			std::cout << "Error! Invalide number for user_method";
			throw(std::runtime_error("user method"));
			break;
		}
		}
	}
	input_method = user_method;
}

void ml::read_img(std::vector<std::string>& dir, holder_M& holder, bool data_type)
{
	fix = false;
	holder = std::make_shared<cv::Mat>(cv::Mat());
	std::shared_ptr<cv::Mat> buffer;
	for (int i = 0; i != dir.size(); i++)
	{
		
		buffer = std::make_shared<cv::Mat>(cv::Mat(cv::imread(dir[i], cv::IMREAD_COLOR)));
		if (fix == false) {
			if ((*buffer).cols * (*buffer).rows >= 750000) {
				cv::resize(*buffer, *buffer, cv::Size(500, 500));
				s = cv::Size(500, 500);
				fix = true;
			}
			else
			{
				if ((*buffer).cols * (*buffer).rows >= 30000) {
					cv::resize(*buffer, *buffer, cv::Size(100, 100));
					s = cv::Size(100, 100);
					fix = true;
				}
				else
				{
					if ((*buffer).cols * (*buffer).rows >= 1200) {
						cv::resize(*buffer, *buffer, cv::Size(20, 20));
						s = cv::Size(20, 20);
						fix = true;
					}
					else
					{
						if ((*buffer).cols * (*buffer).rows >= 75) {
							cv::resize(*buffer, *buffer, cv::Size(5, 5));
							s = cv::Size(5, 5);
							fix = true;
						}
					}
				}
			}
		}
		else 
		{
			cv::resize(*buffer, *buffer, s);
		}
		(*change_input)(buffer);
		cv::Mat RGB[3];
		cv::split(*buffer, RGB);
		std::shared_ptr<cv::Mat> buffer1 = std::make_shared<cv::Mat>(cv::Mat());
		for (int j = 0; j != (*buffer).rows; j++) 
		{
			(*buffer1).push_back((RGB[0]).row(j).t());
			(*buffer1).push_back((RGB[1]).row(j).t());
			(*buffer1).push_back((RGB[2]).row(j).t());
		}
		(*holder).push_back((*buffer1).t());
	}
	(*holder).convertTo(*holder, CV_64FC1);
	if (data_type == 0) {
		if (eigenvector.empty()) {
			data_compression(holder);
		}
		else
		{
			*holder = *holder * (eigenvector).t();
		}
	}
	buffer.reset();
}

void ml::read_txt(std::vector<std::string>& dir, holder_M& holder, bool data_type)
{
	//std::cout << "start reading .txt" << std::endl;
	holder = std::make_shared<cv::Mat>(cv::Mat());
	std::ifstream file(dir[0]);
	if (!file) 
	{

		std::cout << "file " << dir[0] << " not found";
	}
	int current_line = 0;
	std::string line;
	int variable_c;
	while (!file.eof()) {
		while (std::getline(file, line))
		{
			std::istringstream buffer_l(line);
			char dbuffer;
			cv::Mat Mbuffer;
			while (buffer_l >> dbuffer)
			{
				Mbuffer.push_back(double(dbuffer) - 48);
			}
			(*holder).push_back(Mbuffer.t());
			if (current_line == 0)
			{
				variable_c = Mbuffer.rows;
			}
			else
			{
				if (Mbuffer.rows != variable_c)
				{
					std::cout << "samples with different variables. Error occur at line " << current_line;
					throw(std::runtime_error("incorrect data format"));
				}
			}
			current_line++;
		}
	}
	file.close();
	(*change_input)(holder);
	/*
	if (data_type == 0) {
		if (eigenvector.empty()) {
			data_compression(holder);
		}
		else
		{
			*holder = *holder * (eigenvector).t();
		}
	}
	*/

	//std::cout << "finish reading .txt" << std::endl;
	//std::cout << *holder << std::endl;
}

inline void ml::empty_check(std::vector<std::string> V) const
{
	if (V.size() == 0) 
	{
		std::cout << " wrong file name or file type is entered or entered folder entered is empty";
		system("pause");
		throw(std::runtime_error("incorrect folder"));
	}
}

void ml::Ftype_check(std::vector<std::string> V) const
{
	for (auto i : V) 
	{
		if (i.substr(i.size() - file_T.size(), i.size() - 1) != file_T)
		{
			std::cout << "inappropriate files inside selected folder or incorrect file format entered (suffix is required)" ;
			system("pause");
			throw(std::runtime_error("incorrect files"));
		}
	}
}

void ml::check_variance() const
{
	if (variance_maintained > 1 || variance_maintained < 0) 
	{
		std::cout << "variance pass to ml object is out of range";
		system("pause");
		throw(std::runtime_error("variance_out_of_range"));
	}
}

cv::Mat ml::predict(cv::Mat& user_input, int user_method, int output_type, std::vector<double> structure)// 0: simple, 1:kernel, 2: neural network
{
	neuron_structure = structure;
	if ((user_input).cols != (*input).cols) 
	{
		std::cout << "input has the wrong number of variable features";
		throw(std::runtime_error("incorrect test input format"));
	}
	check_null(user_method);
	std::shared_ptr<cv::Mat> buffer = std::make_shared<cv::Mat>(user_input);
	cv::Mat return_buffer = ((*m).calculate(buffer));
	if (output_type == 0) {
		std::cout << "prediction: " << std::endl << test_input_files << std::endl << return_buffer << std::endl;
	}
	else 
	{
		cv::namedWindow("result", cv::WINDOW_NORMAL);
		cv::imshow("result", return_buffer);
		cv::waitKey();
	}/*
	bool save;
	std::cout << "1 to save, 0 to discard";
	std::cin >> save;
	if (save) 
	{
		(*this).save();
	}
	*/
	return return_buffer;
}

cv::Mat ml::predict(std::string&& input_file_dir, std::string&& input_file_type, int user_method, int output_type, std::vector<double> structure)
{
	neuron_structure = structure;
	if (input_file_type == ".txt") 
	{
		test_input_files.push_back(input_file_dir);
		file_T = input_file_type;
		Ftype_check(test_input_files);
		read_txt(test_input_files, test_input, 0);
	}
	else 
	{
		cv::glob(input_file_dir, test_input_files, false);
		//std::cout << test_input_files;
		file_T = input_file_type;
		empty_check(test_input_files);
		Ftype_check(test_input_files);
		read_img(test_input_files, test_input, 0);
	}
	return predict(*test_input, user_method, output_type, neuron_structure);
}
/*
void ml::save()
{
	std::string file_directory;
	std::string file_name;
	std::cout << "input the directory or press enter to save it in the .exe folder" << std::endl;
	std::cin >> file_directory;
	std::cout << "input the file name ( include file type:.txt)" << std::endl;
	std::cin >> file_name;
	if (file_directory.empty()) 
	{
		std::ofstream file(file_name);
	}
	else 
	{
		std::ofstream file(file_directory + file_name);
	}
	
	save file format:
	input_method
	picture size s

	
}
*/

