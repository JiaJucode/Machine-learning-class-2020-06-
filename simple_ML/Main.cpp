#include "ML.h"

void is(holder_M& a)
{
	if ((*a).channels() == 3) {
		cv::cvtColor(*a, *a, cv::COLOR_RGB2HSV);
	}
}

int main()
{
	double tim = cv::getTickCount();
	std::vector<double> st = {5,10};
	void(*n)(holder_M&) = &is;
	ml A("C:/Users/User/Documents/ml_testing/A", ".jpg", "C:/Users/User/Documents/ml_testing/inputA.txt", ".txt", is, 0.99);
	A.predict("C:/Users/User/Documents/ml_testing/test", ".jpg", 2, 0, st);
	A.print_error(2);
	std::cout << "\n total time: " << (cv::getTickCount() - tim) / cv::getTickFrequency() << std::endl;
	return 0;
}