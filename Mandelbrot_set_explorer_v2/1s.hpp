#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace boost::multiprecision;


class point // just a pair of numbers
{
public:
	cpp_dec_float_50 x = 0;
	cpp_dec_float_50 y = 0;

	point set_point(const cpp_dec_float_50& x, const cpp_dec_float_50& y)
	{
		this->x = x;
		this->y = y;

		return *this;
	}
};

class positions // positions and handling coordinates and edge points
{
public:

	point top_left;
	point top_right;
	point down_left;
	point down_right;

	cpp_dec_float_50 step;

	positions(unsigned int WIDTH, unsigned int HEIGHT) // starting positions dependent on Width and Height
	{
		if (WIDTH >= HEIGHT)
		{
			step = 4 / (HEIGHT - 1);

			top_left.y = 2;
			top_left.x = step * ((WIDTH - 1) / 2);
		}
		else
		{
			step = 4 / (WIDTH - 1);

			top_left.x = -2;
			top_left.y = step * ((HEIGHT - 1) / 2);
		}

		down_left.x = top_left.x;
		down_left.y = top_left.y - ((HEIGHT - 1) * step);

		top_right.y = top_left.y;
		top_right.x = top_left.x + ((WIDTH - 1) * step);

		down_right.x = top_right.x;
		down_right.y = down_left.y;

	}

	static void recalculate()
	{

	}
};

class mouse_vars
{
public:
	sf::Vector2i mousePosition;

	bool left_button_down = false;
	bool right_button_down = false;
	bool middle_button_down = false;

	void update(const sf::RenderWindow& window)
	{
		mousePosition = sf::Mouse::getPosition(window);
	}
};

class screen_vars
{
public:
	bool has_focus = true;

	void update(const sf::RenderWindow& window)
	{
		has_focus = window.hasFocus();
	}
};

class variables
{
public:

	mouse_vars MouseVars;
	screen_vars ScreenVars;

	void variables_update(const sf::RenderWindow& window)
	{
		MouseVars.update(window);
		ScreenVars.update(window);
	}
};

inline long long now()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

namespace colorHandling
{
	tuple<int, int, int> numberToRGB(const int& number, const int& iterations) {
		if (number == 0)
		{
			return std::make_tuple(0, 0, 0); // Black color
		}
		else if (number == iterations)
		{
			return std::make_tuple(0, 0, 0); // Black color
		}
		else
		{
			int scaledNumber = (number * 255) / iterations;
			int red = int((255 - scaledNumber) / 1.3) % 256;
			int green = (scaledNumber * 7) % 256;
			int blue = (scaledNumber * 13) % 256;
			return std::make_tuple(red, green, blue);
		}
	}
}

void updatePixels(int* screen, sf::Uint8* pixels, unsigned int WIDTH, unsigned int HEIGHT, const int& iterations)
{
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			int color = screen[y * WIDTH + x];
			auto rgb = colorHandling::numberToRGB(color, iterations);

			int pixelIndex = (x + y * WIDTH) * 4;
			pixels[pixelIndex] = std::get<0>(rgb);
			pixels[pixelIndex + 1] = std::get<1>(rgb);
			pixels[pixelIndex + 2] = std::get<2>(rgb);
			pixels[pixelIndex + 3] = 255;
		}
	}
}
