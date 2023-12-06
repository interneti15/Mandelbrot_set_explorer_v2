#pragma once

#include <mutex>
#include <SFML/Graphics.hpp>

namespace global
{
	extern constexpr unsigned int WIDTH = 900;
	extern constexpr unsigned int HEIGHT = 900;
	extern int max_iterations = 1000;

	extern int* screen;
	extern sf::Uint8* pixels;

	

	extern std::mutex screenMutex;


}
