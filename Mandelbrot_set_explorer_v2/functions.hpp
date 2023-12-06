#pragma once

#include <classes.hpp>

using namespace std;

inline void end()
{
	delete[] global::screen;
	delete[] global::pixels;

	exit(0);
}

inline long long now()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

