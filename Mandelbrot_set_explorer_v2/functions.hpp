#pragma once
#include <tuple>
#include <SFML/Config.hpp>

#include "classes.hpp"

#include <fstream>
#include <mutex>

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

    if (!file.is_open())
    {
        return false;
    }

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
 
    cout << "loaded" << endl;
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
			int color = screen[y * WIDTH + x] - 1;

            //cout << color << endl;
            if (color < 0 && x > 2 && x < WIDTH - 2 && y > 2 && y < HEIGHT - 2)
            {
                color = 0;
                int count = 0;

                for (int i = -1; i <= 1 ; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        if (!i && !j)
                        {
                            continue;
                        }
                        else if (screen[(y + i) * WIDTH + (x + j)] - 1 < 0)
                        {
                            continue;
                        }
                        else
                        {
                            color += screen[(y + i) * WIDTH + (x + j)] - 1;
                            count++;
                        }
                        
                    }
                }
                
                if (count == 0)
                {
                    count = 1;
                }
                color = (int)(color / count);
                
            }

            if (color < 0)
            {
                color = iterations;
            }

			auto rgb = colorHandling::numberToRGB(color, iterations);

			int pixelIndex = (x + y * WIDTH) * 4;
			pixels[pixelIndex] = std::get<0>(rgb);
			pixels[pixelIndex + 1] = std::get<1>(rgb);
			pixels[pixelIndex + 2] = std::get<2>(rgb);
			pixels[pixelIndex + 3] = 255;
		}
	}
}

inline void updatePixels_forThread(int* screen, sf::Uint8* pixels, unsigned int WIDTH, unsigned int HEIGHT, const int& iterations, globals* Global)
{
    while (!Global->Pend)
    {
        for (int y = 0; y < HEIGHT && !Global->Pend; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                int color = screen[y * WIDTH + x] - 1;

                //cout << color << endl;
                if (color < 0 && x > 2 && x < WIDTH - 2 && y > 2 && y < HEIGHT - 2)
                {
                    color = 0;
                    int count = 0;

                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            if (!i && !j)
                            {
                                continue;
                            }
                            else if (screen[(y + i) * WIDTH + (x + j)] - 1 < 0)
                            {
                                continue;
                            }
                            else
                            {
                                color += screen[(y + i) * WIDTH + (x + j)] - 1;
                                count++;
                            }

                        }
                    }

                    if (count == 0)
                    {
                        count = 1;
                    }
                    color = (int)(color / count);

                }

                if (color < 0)
                {
                    color = iterations;
                }

                auto rgb = colorHandling::numberToRGB(color, iterations);

                
                int pixelIndex = (x + y * WIDTH) * 4;
                Global->pixelMutex.lock();
                pixels[pixelIndex] = std::get<0>(rgb);
                pixels[pixelIndex + 1] = std::get<1>(rgb);
                pixels[pixelIndex + 2] = std::get<2>(rgb);
                pixels[pixelIndex + 3] = 255;
                Global->pixelMutex.unlock();
            }
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

/*

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

inline void testt(globals* Global, const positions& cords)
{
    cout << "started" << endl;

    cout << "X: " << cords.top_left.x << " Y: " << cords.top_left.y << " S: " << cords.step << endl;

	const int height = Global->HEIGHT;
    const int width = Global->WIDTH;

	for (size_t y = 0; y < height && !(Global->end); y++)
    {       
        for (size_t x = 0; x < width && !(Global->end); x++)      
        {
            if (Global->screen[x + width * y] != 0)
            {
                continue;
            }

	        const cpp_dec_float_50 xp = cords.top_left.x + x * cords.step;
	        const cpp_dec_float_50 yp = cords.top_left.y - y * cords.step;

	        const int re = iteration_check(xp, yp, Global->max_iterations);

            Global->screenMutex.lock();
            Global->screen[x + y * width] = re + 1;
            Global->screenMutex.unlock();
        }
    }

    cout << "ended" << endl;
}

*/

inline void moveScreen(int Dx, int Dy, globals* Globals)
{
    int* screen = new int[Globals->WIDTH * Globals->HEIGHT];

    const int height = Globals->HEIGHT;
    const int width = Globals->WIDTH;

    size_t x,y;

    if (Dy < 0)
        y = 0;
    else 
        y = Dy;

    while (y < height && y < height + Dy)
    {
        if (Dx < 0) 
            x = 0;
        else
            x = Dx;

        while (x < width && x < width + Dx)
        {
            screen[x + y * width] = Globals->screen[(x - Dx) + (y - Dy) * width];
            x++;
        }

        y++;
    }

    Globals->screen = screen;
}

inline void cleanScreen(globals* Globals)
{
    Globals->screen = new int[Globals->WIDTH * Globals->HEIGHT];
}



