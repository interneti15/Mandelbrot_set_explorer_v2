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

void end(globals& Global ,const int& code, threadsHandling& Threads, thread* SC)
{
	Threads.killAll(&Global);

	Global.Pend = true;
	SC->join();

	delete[] Global.screen;
	delete[] Global.pixels;

	exit(code);
}

int main()
{
	globals Global;
	Global.clean();

	cout << "Resolution: " << Global.HEIGHT << "X" << Global.WIDTH << endl;

	sf::RenderWindow window(sf::VideoMode(Global.WIDTH, Global.HEIGHT), "Mandelbrot Set", sf::Style::Titlebar | sf::Style::Close);

	sf::Texture texture;
	texture.create(Global.WIDTH, Global.HEIGHT);

	sf::Sprite sprite(texture);

	variables vars;

	positions cords(Global.WIDTH, Global.HEIGHT);

	threadsHandling Threads(&cords, &Global);

	thread Sc_update(updatePixels_forThread, Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations, &Global);

	screenText screentext;


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

			vars.ScreenVars.lastposition.set(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);

		}
		if (vars.MouseVars.left_button_down && vars.ScreenVars.after_grab && vars.ScreenVars.has_focus)
		{
			

			if (pointsDistance(vars.ScreenVars.lastposition, intPoint(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y), 30))
			{
				vars.ScreenVars.lastposition.set(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);

				Threads.killAll(&Global);

				sprite.setPosition(0, 0);

				int dx = (int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x);
				int dy = (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y);

				Global.Pend = true;
				Sc_update.join();
				Global.Pend = false;

				moveScreen(dx, dy, &Global);
				cords.recalculate(dx, dy);

				Sc_update = thread(updatePixels_forThread, Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations, &Global);

				Threads.start(&cords, &Global);

				vars.ScreenVars.grab_point.set_point(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);
			}
			else
			{
				sprite.setPosition((int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x), (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y));
			}
			
			
		}

		if (!vars.MouseVars.left_button_down && vars.ScreenVars.after_grab && vars.ScreenVars.has_focus)
		{
			Threads.killAll(&Global);

			vars.ScreenVars.after_grab = false;

			sprite.setPosition(0, 0);

			int dx = (int)(vars.MouseVars.mousePosition.x - vars.ScreenVars.grab_point.x);
			int dy = (int)(vars.MouseVars.mousePosition.y - vars.ScreenVars.grab_point.y);

			Global.Pend = true;
			Sc_update.join();
			Global.Pend = false;
		
			moveScreen(dx, dy, &Global);
			cords.recalculate(dx, dy);
					
			Sc_update = thread(updatePixels_forThread, Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations, &Global);
			
			Threads.start(&cords, &Global);
		}
		

		if ((int)(event.mouseWheelScroll.delta) > 0 && vars.ScreenVars.has_focus && (vars.MouseVars.mousePosition.x >= 0 && vars.MouseVars.mousePosition.x < Global.WIDTH && vars.MouseVars.mousePosition.y >= 0 && vars.MouseVars.mousePosition.y < Global.HEIGHT))
		{
			Threads.killAll(&Global);

			Global.Pend = true;
			Sc_update.join();
			Global.Pend = false;

			//cleanScreen(&Global);
			zoom_in(&Global, &vars);

			cords.zoom_in(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);

			cout << "Zin: " << cords.top_left.x << " : " << cords.top_left.y << endl;

			Threads.start(&cords, &Global);

			event.mouseWheelScroll.delta = 0;

			Sc_update = thread(updatePixels_forThread, Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations, &Global);
		}
		else if ((int)(event.mouseWheelScroll.delta) < 0 && vars.ScreenVars.has_focus && (vars.MouseVars.mousePosition.x >= 0 && vars.MouseVars.mousePosition.x < Global.WIDTH && vars.MouseVars.mousePosition.y >= 0 && vars.MouseVars.mousePosition.y < Global.HEIGHT))
		{
			Threads.killAll(&Global);

			Global.Pend = true;
			Sc_update.join();
			Global.Pend = false;

			//cleanScreen(&Global);
			zoom_out(&Global, &vars);

			cords.zoom_out(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y);

			cout << "Zout: " << cords.top_left.x << " : " << cords.top_left.y << endl;

			Threads.start(&cords, &Global);

			event.mouseWheelScroll.delta = 0;

			Sc_update = thread(updatePixels_forThread, Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations, &Global);
		}
		
		if (true)
		{
			//updatePixels(Global.screen, Global.pixels, Global.WIDTH, Global.HEIGHT, Global.max_iterations);

			Global.pixelMutex.lock();
			texture.update(Global.pixels);
			Global.pixelMutex.unlock();

			window.clear();

			window.draw(sprite);
			screentext.refresh(vars.MouseVars.mousePosition.x, vars.MouseVars.mousePosition.y, cords, Global, window);
		}

		window.display();

		if (vars.MouseVars.right_button_down)
		{
			
			//cout << Global.screen[vars.MouseVars.mousePosition.x + Global.WIDTH * vars.MouseVars.mousePosition.y] << endl;
			
			
		}

	}
	end(Global,1, Threads, &Sc_update);
}

