#pragma once
#include <iostream>
#include <stdio.h>

namespace cppPrNs
{

    class precisionNumber
    {
    private:
        int precision;
        int* top;
        int* bottom;

    public:
        precisionNumber(const int desiredPrecision = 50) : precision(desiredPrecision) 
        {
            top = new int[desiredPrecision];
            bottom = new int[desiredPrecision];
        }

	};


}