#include <thread>
#include <iostream>

#include <SFML/Graphics.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include "classes.cpp"

#include <mutex>


using namespace boost::multiprecision;
using namespace std;

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

int main()
{
	sf::RenderWindow window(sf::VideoMode(global::WIDTH, global::HEIGHT), "Mandelbrot Set", sf::Style::Titlebar | sf::Style::Close);

	sf::Texture texture;
	texture.create(global::WIDTH, global::HEIGHT);

	sf::Sprite sprite(texture);

	sf::Font font;
	font.loadFromFile("arial.ttf");

	sf::Text text("", font, 25);
	text.setOutlineColor(sf::Color::Black);
	text.setOutlineThickness(1);

	variables vars;

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

		if (true)
		{
			updatePixels(global::screen, global::pixels, global::WIDTH, global::HEIGHT, global::max_iterations);

			texture.update(global::pixels);

			window.clear();
			window.draw(sprite);
		}

		window.display();
	}
	//end();
}

