#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <SFML/Graphics.hpp>


using namespace std;
using namespace boost::multiprecision;

class globals
{
public:
	const unsigned int WIDTH = 1200;
	const unsigned int HEIGHT = 900;

	int* screen = new int[WIDTH * HEIGHT];
	sf::Uint8* pixels = new sf::Uint8[WIDTH * HEIGHT * 4];

	int max_iterations = 20;

	bool end = false;

	std::mutex screenMutex;

};

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

	cpp_dec_float_50 step;

	positions(cpp_dec_float_50 WIDTH, cpp_dec_float_50 HEIGHT) // starting positions dependent on Width and Height
	{
		if (WIDTH >= HEIGHT)
		{
			step = 4 / (HEIGHT - 1);

			top_left.y = 2;
			top_left.x = -(step * ((WIDTH - 1) / 2));
		}
		else
		{
			step = 4 / (WIDTH - 1);

			top_left.x = -2;
			top_left.y = step * ((HEIGHT - 1) / 2);
		}

	}

	void recalculate(const int Dx, const int Dy)
	{
		//cout << "Dx: " << Dx << " Dy: " << Dy << endl;

		this->top_left.x -= (Dx * step);
		this->top_left.y += (Dy * step);
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

		left_button_down = sf::Mouse::isButtonPressed(sf::Mouse::Left);
		right_button_down = sf::Mouse::isButtonPressed(sf::Mouse::Right);
		middle_button_down = sf::Mouse::isButtonPressed(sf::Mouse::Middle);
	}
};

class screen_vars
{
public:
	bool has_focus = true;
	bool after_grab = false;
	point grab_point;

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




