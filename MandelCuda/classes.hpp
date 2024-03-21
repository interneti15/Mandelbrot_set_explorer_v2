#pragma once

#include "cppPrecisionNums.hpp"

#include <SFML/Graphics.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <iostream>


using namespace std;
using namespace boost::multiprecision;

class globals
{
public:
	const unsigned int WIDTH = 1200;
	const unsigned int HEIGHT = 900;

	int* screen = new int[WIDTH * HEIGHT]();
	int* history = new int[WIDTH * HEIGHT]();
	sf::Uint8* pixels = new sf::Uint8[WIDTH * HEIGHT * 4]();

	bool GpuAcceleration = false;

	//int* screen = static_cast<int*>(calloc(WIDTH * HEIGHT, sizeof(int)));
	//int* history = static_cast<int*>(calloc(WIDTH * HEIGHT, sizeof(int)));
	//sf::Uint8* pixels = static_cast<sf::Uint8*>(calloc(WIDTH * HEIGHT * 4, sizeof(sf::Uint8)));

	const unsigned max_iterations = 1000;

	std::mutex screenMutex;
	const unsigned int CPU_threads = std::thread::hardware_concurrency();
	//const unsigned int CPU_threads = 1;

	bool end = false;
	bool Pend = false;

	mutex pixelMutex;

	void clean()
	{
		std::fill(screen, screen + WIDTH * HEIGHT, 0);
		std::fill(history, history + WIDTH * HEIGHT, 0);
	}

	void historyReset()
	{
		std::fill(history, history + WIDTH * HEIGHT, 0);
	}

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

	cpp_dec_float_50 starting_size;

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

		starting_size = HEIGHT * WIDTH * step * step;
	}

	void recalculate(const int Dx, const int Dy)
	{
		//cout << "Dx: " << Dx << " Dy: " << Dy << endl;

		top_left.x -= (Dx * step);
		top_left.y += (Dy * step);
	}

	void zoom_in(int x, int y)
	{
		const cpp_dec_float_50 xP = top_left.x + (x * step);
		const cpp_dec_float_50 yP = top_left.y - (y * step);

		cout << "xP: " << xP << " yP: " << yP << endl;

		step = step / 2;

		top_left.x = xP - (x * step);
		top_left.y = yP + (y * step);
	}

	void zoom_out(int x, int y)
	{
		const cpp_dec_float_50 xP = top_left.x + (x * step);
		const cpp_dec_float_50 yP = top_left.y - (y * step);

		cout << "xP: " << xP << " yP: " << yP << endl;

		step = step * 2;

		top_left.x = xP - (x * step);
		top_left.y = yP + (y * step);
	}
};

class intPoint
{
public:

	int x;
	int y;

	intPoint(int x = 0, int y = 0)
	{
		this->x = x;
		this->y = y;
	}

	void set(int x, int y)
	{
		this->x = x;
		this->y = y;
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

	intPoint lastposition;

	bool scroll_swich_up = true;
	bool scroll_swich_down = true;

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



class threadsHandling
{
public:
	vector<thread> list_of_threads;
	vector<intPoint> preparedCords;

	void prepare(positions* pos, globals* Global, bool newS)
	{

		const unsigned int height = Global->HEIGHT;
		const unsigned int width = Global->WIDTH;

		vector<intPoint> pCords;

		if (!newS || true)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					//cout << Global->screen[x + y * width] << endl;
					if (Global->screen[x + y * width] == 0)
					{
						pCords.push_back(intPoint(x, y));
					}
				}
			}
		}

		preparedCords.clear();

		std::random_shuffle(pCords.begin(), pCords.end());

		preparedCords = pCords;
		//cout << "rand" << endl;
	}

	void killAll(globals* Global)
	{
		Global->end = true;

		for (size_t i = 0; i < list_of_threads.size(); i++)
		{
			list_of_threads[i].join();
		}

		Global->end = false;
	}

	threadsHandling(positions* pos, globals* Global, bool newS = false)
	{
		this->prepare(pos, Global, newS);

		int chunk = (int)((preparedCords.size()) / Global->CPU_threads);

		list_of_threads = vector<thread>(Global->CPU_threads);

		//killAll(Global);

		for (size_t i = 0; i < Global->CPU_threads; i++)
		{
			if (i == Global->CPU_threads - 1)
			{
				list_of_threads[i] = thread(calculatorFunction, i * chunk, preparedCords.size() - 1, Global, pos, preparedCords);
				continue;
			}
			list_of_threads[i] = thread(calculatorFunction, i * chunk, (i + 1) * chunk - 1, Global, pos, preparedCords);

		}
	}

	void start(positions* pos, globals* Global, bool newS = false)
	{
		this->prepare(pos, Global, newS);

		int chunk = (int)((preparedCords.size()) / Global->CPU_threads);

		list_of_threads = vector<thread>(Global->CPU_threads);

		for (size_t i = 0; i < Global->CPU_threads; i++)
		{
			if (i == Global->CPU_threads - 1)
			{
				list_of_threads[i] = thread(calculatorFunction, i * chunk, preparedCords.size() - 1, Global, pos, preparedCords);
				continue;
			}
			list_of_threads[i] = thread(calculatorFunction, i * chunk, (i + 1) * chunk - 1, Global, pos, preparedCords);

		}
	}

private:
	static bool from0(const cpp_dec_float_50& x, const cpp_dec_float_50& y, const cpp_dec_float_50& len = 50)
	{
		return ((x * x) + (y * y)) >= len * len;
	}

	static int iteration_check(cpp_dec_float_50 x, cpp_dec_float_50 y, const int& max_iterations)
	{
		const cpp_dec_float_50 x0 = x;
		const cpp_dec_float_50 y0 = y;

		for (int i = 0; i < max_iterations; ++i)
		{
			if (from0(x, y))
			{
				return i;
			}

			const cpp_dec_float_50 x_temp = x * x - y * y + x0;
			y = 2 * x * y + y0;
			x = x_temp;
		}

		return max_iterations;
	}

	static void calculatorFunction(int starting_index, int ending_index, globals* Global, positions* cords, const vector<intPoint>& preparedCords)
	{
		const int height = Global->HEIGHT;
		const int width = Global->WIDTH;

		if (preparedCords.empty())
		{
			return;
		}

		for (size_t i = starting_index; i <= ending_index && !(Global->end); i++)
		{
			const int x = preparedCords[i].x;
			const int y = preparedCords[i].y;

			const cpp_dec_float_50 xp = cords->top_left.x + x * cords->step;
			const cpp_dec_float_50 yp = cords->top_left.y - y * cords->step;

			const int re = iteration_check(xp, yp, Global->max_iterations);

			//cout << x << " : " << y << " : " << re << endl;

			//Global->screenMutex.lock();
			Global->screen[x + y * width] = re + 1;
			//Global->screenMutex.unlock();
		}
	}
};

class screenText
{
public:
	sf::Font font;

	sf::Text xText;
	sf::Text yText;
	sf::Text zoomText;


	screenText()
	{
		font.loadFromFile("arial.ttf");

		xText = sf::Text("", font, 15);
		xText.setOutlineColor(sf::Color::Black);
		xText.setOutlineThickness(1);
		xText.setPosition(5, 20);

		yText = sf::Text("", font, 15);
		yText.setOutlineColor(sf::Color::Black);
		yText.setOutlineThickness(1);
		yText.setPosition(5, 40);

		zoomText = sf::Text("", font, 15);
		zoomText.setOutlineColor(sf::Color::Black);
		zoomText.setOutlineThickness(1);
		zoomText.setPosition(5, 60);
	}

	void refresh(const int x, const int y, positions& cords, globals& Global, sf::RenderWindow& window)
	{

		const cpp_dec_float_50 zoom = cords.starting_size / (cords.step * Global.HEIGHT * cords.step * Global.HEIGHT);
		string zoomT = to_string(zoom), zT;
		zT += zoomT[0];
		zT += ".";
		zT += zoomT[2];
		zT += "e";
		zT += to_string(static_cast<int>(log10(zoom)));
		zoomText.setString("zoom: " + zT);

		std::ostringstream xT, yT;
		if (!(x < 0 || y < 0 || x > Global.WIDTH - 1 || y > Global.HEIGHT - 1) || (static_cast<int>(log10(zoom))) < 15)
		{
			const cpp_dec_float_50 xP = cords.top_left.x + (x * cords.step);
			const cpp_dec_float_50 yP = cords.top_left.y - (y * cords.step);

			xT << "x: " << xP;
			yT << "y: " << yP;
		}
		else
		{
			xT << "x: NaN";
			yT << "y: NaN";
		}
		//cout << "xP: " << xP << " yP: " << yP << endl;

		xText.setString(xT.str());
		yText.setString(yT.str());



		window.draw(xText);
		window.draw(yText);
		window.draw(zoomText);
	}


private:

};


