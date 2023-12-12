#pragma once
#include <tuple>
#include <SFML/Config.hpp>

#include "classes.hpp"

#include <fstream>

inline long long now()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline bool load(int* screen, int width, int height)
{
    ifstream file;
    string path = to_string(width) + 'X' + to_string(height) + ".txt";

    file.open(path);

    string data, temp;

    

    getline(file, data);

    int j = 0;
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i] == ';')
        {
            screen[j] = stoi(temp);
            j++;
            temp.clear();
            continue;
        }
        temp += data[i];
    }
    screen[j] = stoi(temp);
 
    cout << 1;
}

namespace colorHandling
{
	inline std::tuple<int, int, int> numberToRGB(const int& number, const int& iterations) {
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

inline void updatePixels(int* screen, sf::Uint8* pixels, unsigned int WIDTH, unsigned int HEIGHT, const int& iterations)
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

inline void paint(int* screen, const variables vars, int width, int height)
{
	if (vars.MouseVars.mousePosition.x >= 0 && vars.MouseVars.mousePosition.y >= 0 && vars.MouseVars.mousePosition.x < width && vars.MouseVars.mousePosition.y < height)
	{
		screen[vars.MouseVars.mousePosition.x + width * vars.MouseVars.mousePosition.y] = 500;
	}
}

inline bool from0(const cpp_dec_float_50& x, const cpp_dec_float_50& y, const cpp_dec_float_50& len = 50) 
{
    return ((x * x) + (y * y)) >= len * len;
}

inline int iteration_check(cpp_dec_float_50 x, cpp_dec_float_50 y, const int& max_iterations) 
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

inline void painttest(int* screen, const int& width, const int& height, const positions& cords)
{
    cout << cords.top_left.x << endl;
    cout << cords.top_left.y << endl;

    for (size_t y = 0; y < height; y++)
    {       
        for (size_t x = 0; x < width; x++)
        {
	        const cpp_dec_float_50 xp = cords.top_left.x + x * cords.step;
	        const cpp_dec_float_50 yp = cords.top_left.y - y * cords.step;

            //cout << "X: " << x << " : " << xp << "    Y: " << y << " : " << yp << endl;

            screen[x + y * width] = iteration_check(xp, yp, 1000);
        }
    }
}

/*
void calculateMandelbrot(int startRow, int endRow, int i, bool dry_run) {

    if (dry_run)
    {
        is_alive[i] = false;
        return;
    }

    is_alive[i] = true;
    to_clean = true;

    int last_color = 0, color = 0;
    bool first = true, change = false;
    int x = 0, temp_color = 0;
    cpp_dec_float_50 y_loc = 0, x_loc = 0;

    cout << "Started: " << i << endl;

    if (more_details)
    {
        for (int y = startRow; y < endRow and !end_all_threads; y++) {
            y_loc = maxy - (ystep * y) - (ystep / 2);

            int x = 0;

            for (int x = 0; x < WIDTH and !end_all_threads; x++) {
                x_loc = minx + (xstep * x) + (xstep / 2);

                color = iteration_check(x_loc, y_loc);

                screenMutex.lock();
                screen[y * WIDTH + x] = color;
                screenMutex.unlock();
            }
        }
    }
    else
    {
        for (int y = startRow; y < endRow and !end_all_threads; y++) {
            y_loc = maxy - (ystep * y);

            x = 0;
            x_loc = minx + (xstep * x);

            color = iteration_check(x_loc, y_loc);

            screenMutex.lock();
            screen[y * WIDTH + x] = color;
            screenMutex.unlock();

            last_color = color;

            x += 2;
            while (x < WIDTH and !end_all_threads) {
                x_loc = minx + (xstep * x);

                color = iteration_check(x_loc, y_loc);

                screenMutex.lock();
                screen[y * WIDTH + x] = color;
                screenMutex.unlock();

                if (change)
                {
                    change = false;
                    x += 1;
                    color = temp_color;
                }
                else if (last_color != color)
                {
                    temp_color = color;
                    x -= 1;
                    change = true;
                    continue;
                }
                else if (last_color == color)
                {
                    screenMutex.lock();
                    screen[y * WIDTH + (x - 1)] = color;
                    screenMutex.unlock();
                }

                if (x + 2 < HEIGHT - 1)
                {
                    x += 2;
                }
                else
                {
                    x += 1;
                }
                last_color = color;
            }
            //cout << y << endl;
        }
    }
    cout << "Ended: " << i << endl;
    is_alive[i] = false;
}
*/
