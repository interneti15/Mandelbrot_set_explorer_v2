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

void end(const globals& Global ,const int& code = 0)
{
	delete[] Global.screen;
	delete[] Global.pixels;

	exit(code);
}

int main()
{
	globals Global;

	sf::RenderWindow window(sf::VideoMode(Global.WIDTH, Global.HEIGHT), "Mandelbrot Set", sf::Style::Titlebar | sf::Style::Close);

	sf::Texture texture;
	texture.create(Global.WIDTH, Global.HEIGHT);

	sf::Sprite sprite(texture);

	variables vars;

	load(Global.screen, Global.WIDTH, Global.HEIGHT);

	positions cords(Global.WIDTH, Global.HEIGHT);

	thread test(testt, &Global, cords);

	//testt(&Global, cords);

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
		if (vars.MouseVars.left_button_down && !vars.ScreenVars.after_grab && vars.ScreenVars.has_focus)
		{
			//paint(Global.screen, vars, Global.WIDTH, Global.HEIGHT);

			vars.ScreenVars.grab_point.set_point(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);
			vars.ScreenVars.after_grab = true;

			Global.end = true;
			test.join();
			Global.end = false;
		}
		if (vars.MouseVars.left_button_down && vars.ScreenVars.after_grab && vars.ScreenVars.has_focus)
		{
			sprite.setPosition((int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x), (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y));
			
		}
		if (!vars.MouseVars.left_button_down && vars.ScreenVars.after_grab && vars.ScreenVars.has_focus)
		{
			vars.ScreenVars.after_grab = false;

			sprite.setPosition(0, 0);
			moveScreen((int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x), (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y), &Global);

			cords.recalculate((int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x), (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y));

			test = thread(testt, &Global, cords);
		}

		if (true)
		{
			updatePixels(Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations);

			texture.update(Global.pixels);

			window.clear();
			window.draw(sprite);
		}

		window.display();

	}
	end(Global,1);
}

