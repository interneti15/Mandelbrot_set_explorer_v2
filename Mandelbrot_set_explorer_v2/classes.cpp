#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace boost::multiprecision;

namespace global
{
	constexpr unsigned int WIDTH = 900;
	constexpr unsigned int HEIGHT = 900;

	int* screen = new int[WIDTH * HEIGHT];
	sf::Uint8* pixels = new sf::Uint8[WIDTH * HEIGHT * 4];

	int max_iterations = 1000;

	std::mutex screenMutex;

}

inline void end()
{
	delete[] global::screen;
	delete[] global::pixels;

	exit(0);
}

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

	positions() // starting positions dependent on Width and Height
	{
		if (global::WIDTH >= global::HEIGHT)
		{
			step = 4 / (global::HEIGHT - 1);

			top_left.y = 2;
			top_left.x = step * ((global::WIDTH - 1) / 2);
		}
		else
		{
			step = 4 / (global::WIDTH - 1);

			top_left.x = -2;
			top_left.y = step * ((global::HEIGHT - 1) / 2);
		}

		down_left.x = top_left.x;
		down_left.y = top_left.y - ((global::HEIGHT - 1) * step);

		top_right.y = top_left.y;
		top_right.x = top_left.x + ((global::WIDTH - 1) * step);

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


