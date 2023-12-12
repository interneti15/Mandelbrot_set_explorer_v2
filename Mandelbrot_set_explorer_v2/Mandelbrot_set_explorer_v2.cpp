#include <thread>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

//#include "classes.hpp"
#include "functions.hpp"

#include <mutex>


using namespace boost::multiprecision;
using namespace std;

namespace global
{
	constexpr unsigned int WIDTH = 1200;
	constexpr unsigned int HEIGHT = 900;

	int* screen = new int[WIDTH * HEIGHT];
	sf::Uint8* pixels = new sf::Uint8[WIDTH * HEIGHT * 4];

	int max_iterations = 1000;

	std::mutex screenMutex;

}

void end(const int& code = 0)
{
	delete[] global::screen;
	delete[] global::pixels;

	exit(code);
}

int main()
{
	sf::RenderWindow window(sf::VideoMode(global::WIDTH, global::HEIGHT), "Mandelbrot Set", sf::Style::Titlebar | sf::Style::Close);

	sf::Texture texture;
	texture.create(global::WIDTH, global::HEIGHT);

	sf::Sprite sprite(texture);

	variables vars;

	load(global::screen, global::WIDTH, global::HEIGHT);

	positions cords(global::WIDTH, global::HEIGHT);

	//cout << cords.step;

	thread test(painttest,global::screen, global::WIDTH, global::HEIGHT,cords);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event)) 
		{
			if (event.type == sf::Event::Closed) 
			{
				window.close();
			}
		}

		vars.variables_update(window);
		if (vars.MouseVars.left_button_down)
		{
			paint(global::screen, vars, global::WIDTH, global::HEIGHT);
		}
		if (vars.MouseVars.right_button_down)
		{
			
		}

		if (true)
		{
			updatePixels(global::screen, global::pixels, global::WIDTH, global::HEIGHT, global::max_iterations);

			texture.update(global::pixels);

			window.clear();
			window.draw(sprite);
		}

		window.display();

	}
	end(1);
}

