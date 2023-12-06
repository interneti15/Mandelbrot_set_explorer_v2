#include <thread>
#include <iostream>
#include <vector>

#include <SFML/Graphics.hpp>
#include <mutex>
#include <cmath>

#include <chrono>

#include <classes.hpp>
#include <functions.hpp>

using namespace boost::multiprecision;
using namespace std;

int main()
{
	sf::RenderWindow window(sf::VideoMode(global::WIDTH, global::HEIGHT), "Mandelbrot Set", sf::Style::Titlebar | sf::Style::Close);

	sf::Texture texture;
	texture.create(global::WIDTH, global::HEIGHT);

	sf::Sprite sprite(texture);

	sf::Font font;
	font.loadFromFile("arial.ttf");

	//sf::Text text(std::string("Zoom: " + maths + "X"), font, 25);
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










	}
}

