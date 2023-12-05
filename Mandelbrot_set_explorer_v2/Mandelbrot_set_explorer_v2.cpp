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
		size_t t_size = 0;
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

			for (size_t i = t_size; i >= 0; i--)
			{
				data_pre_point.push_back((cother %(((int)pow(10, i)))));
			}

		}
		
		//return pow(10, t_size);
		
	}

protected:

	bool sign = true;
	vector<char> data_pre_point;
	vector<char> data_afer_point;

};


int main()
{
	BigFloat a;

	
}

