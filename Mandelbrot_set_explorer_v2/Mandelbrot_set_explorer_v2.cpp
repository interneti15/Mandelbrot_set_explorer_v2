#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class BigFloat
{
public:

	void opt_copy(BigFloat other)
	{
		this->data_afer_point = other.data_afer_point;
		this->data_pre_point = other.data_pre_point;
		this->sign = other.sign;
	}
	
	int operator=(long double other)
	{
		int cother = (int)other;
		int temp_num;
		size_t t_size = 0;

		if (other < 0)
		{
			sign = false;
		}

		if (cother > 1)
		{
			for (size_t i = 1;; i++)
			{
				if (pow(10,i) > cother)
				{
					t_size = i - 1;
					break;
				}
			}

			for (int i = t_size; i >= 0; i -= 1)
			{
				temp_num = cother / (((int)pow(10, i)));
				data_pre_point.push_back((temp_num));
				cother -= temp_num * pow(10, i);
				other -= temp_num * pow(10, i);
			}
		}

		for (size_t i = 0;; i--)
		{

		}
		
	}

	vector<int> f1() { return data_pre_point; }

protected:

	bool sign = true;
	vector<int> data_pre_point;
	vector<int> data_afer_point;

};


int main()
{
	BigFloat a;
	a = 123;

	vector<int> ab = a.f1();

	for (size_t i = 0; i < ab.size(); i++)
	{
		cout << ab[i];
	}
	
}

