#pragma once

#include <chrono>
#include <iostream>
#include <tuple>

#include <SFML/Graphics.hpp>

using namespace std;

inline long long now()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

namespace colorHandling
{
    std::tuple<int, int, int> numberToRGB(const int& number, const int& iterations) {
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