#include <thread>
#include <iostream>
#include <vector>

#include <SFML/Graphics.hpp>
#include <mutex>
#include <cmath>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;
using namespace std;

constexpr int WIDTH = 900;
constexpr int HEIGHT = 900;

std::vector<int> screen(WIDTH* HEIGHT, 0);
sf::Uint8* pixels = new sf::Uint8[WIDTH * HEIGHT * 4];

std::mutex screenMutex;

class Positions
{
public:
	cpp_dec_float_50 x_middle = 0;
	cpp_dec_float_50 y_middle = 0;

	cpp_dec_float_50 Top_Left = 0;
	cpp_dec_float_50 Top_right = 0;

	cpp_dec_float_50 Down_Left = 0;
	cpp_dec_float_50 Down_Right = 0;

	static void recalculate()
	{
		
	}
};

int main()
{

}

