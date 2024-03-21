#pragma once
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

namespace myNumLib
{
    const unsigned char UNSIGNED_CHAR_MAX = 255;

    class bigInt 
    {
    private:

        

    public:
        unsigned char* number;
        int SIZE;

        static bigInt bigIntConstructor(const int MAX_SIZE = 50) {
            bigInt toReturn;

            toReturn.number = new unsigned char[MAX_SIZE]();
            toReturn.SIZE = MAX_SIZE;

            return toReturn;
        }

        __device__ static bigInt deviceBigIntConstructor(const int MAX_SIZE = 50) {
            bigInt toReturn;

            toReturn.number = new unsigned char[MAX_SIZE]();
            toReturn.SIZE = MAX_SIZE;

            return toReturn;
        }

        static bool isFirstBiggerThenSecond(const bigInt a, const bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;

            for (int i = MAX_SIZE - 1; i >= 0; i--)
            {
                if (i < a.SIZE)
                {
                    if (i < b.SIZE)
                    {
                        if (a.number[i] > b.number[i])
                        {
                            return true;
                        }
                        else if(a.number[i] < b.number[i])
                        {
                            return false;
                        }
                    }
                    else if (a.number[i] != 0)
                    {
                        return true;
                    }
                }
                else if (i < b.SIZE)
                {
                    if (i < a.SIZE)
                    {
                        if (a.number[i] > b.number[i])
                        {
                            return true;
                        }
                        else if (a.number[i] < b.number[i])
                        {
                            return false;
                        }
                    }
                    else if (b.number[i] != 0)
                    {
                        return false;
                    }
                }
            }

        }

        __device__ static bool deviceisFirstBiggerThenSecond(const bigInt a, const bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;

            for (int i = MAX_SIZE - 1; i >= 0; i--)
            {
                if (i < a.SIZE)
                {
                    if (i < b.SIZE)
                    {
                        if (a.number[i] > b.number[i])
                        {
                            return true;
                        }
                        else if (a.number[i] < b.number[i])
                        {
                            return false;
                        }
                    }
                    else if (a.number[i] != 0)
                    {
                        return true;
                    }
                }
                else if (i < b.SIZE)
                {
                    if (i < a.SIZE)
                    {
                        if (a.number[i] > b.number[i])
                        {
                            return true;
                        }
                        else if (a.number[i] < b.number[i])
                        {
                            return false;
                        }
                    }
                    else if (b.number[i] != 0)
                    {
                        return false;
                    }
                }
            }

        }

        static bigInt extendMAX_SIZE(const bigInt toExtend, const int MAX_SIZE) {
            bigInt toReturn;
            toReturn.bigIntConstructor(MAX_SIZE);

            for (size_t i = 0; i < toExtend.SIZE; i++)
            {
                toReturn.number[i] = toExtend.number[i];
            }

            return toReturn;
        }

        __device__ static bigInt deviceExtendMAX_SIZE(const bigInt toExtend, const int MAX_SIZE) {
            bigInt toReturn;
            toReturn.deviceBigIntConstructor(MAX_SIZE);

            for (size_t i = 0; i < toExtend.SIZE; i++)
            {
                toReturn.number[i] = toExtend.number[i];
            }

            return toReturn;
        }

        static bigInt add(bigInt a, bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;

            if (a.SIZE > b.SIZE)
            {
                a = extendMAX_SIZE(a, b.SIZE);
            }
            else if (b.SIZE > a.SIZE)
            {
                b = extendMAX_SIZE(b, a.SIZE);
            }

            bigInt toReturn = bigIntConstructor(MAX_SIZE);
            unsigned char toAddNext = 0;

            do
            {
                for (size_t i = 0; i < MAX_SIZE; i++)
                {
                    if (a.number[i] + b.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = a.number[i] + b.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = a.number[i] + b.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }

            } while (toAddNext);

            return toReturn;
        }

        __device__ static bigInt deviceAdd(bigInt a, bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;

            if (a.SIZE > b.SIZE)
            {
                a = deviceExtendMAX_SIZE(a, b.SIZE);
            }
            else if (b.SIZE > a.SIZE)
            {
                b = deviceExtendMAX_SIZE(b, a.SIZE);
            }

            bigInt toReturn = deviceBigIntConstructor(MAX_SIZE);
            unsigned char toAddNext = 0;

            do
            {
                for (size_t i = 0; i < MAX_SIZE; i++)
                {
                    if (a.number[i] + b.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = a.number[i] + b.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = a.number[i] + b.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }

            } while (toAddNext);

            return toReturn;
        }

    };

    class precisionNumber
    {
    private:
        
        

    public:
        int precision;
        bigInt top;
        bigInt bottom;


        static precisionNumber precisionNumberConstructor(const int desiredPrecision = 50)
        {
            precisionNumber toReturn;

            toReturn.top = bigInt::bigIntConstructor(desiredPrecision);
            toReturn.bottom = bigInt::bigIntConstructor(desiredPrecision);
            toReturn.precision = desiredPrecision;
        
            return toReturn;
        }

        __device__ static precisionNumber devicePrecisionNumberConstructor(const int desiredPrecision = 50)
        {
            precisionNumber toReturn;

            toReturn.top = bigInt::deviceBigIntConstructor(desiredPrecision);
            toReturn.bottom = bigInt::deviceBigIntConstructor(desiredPrecision);
            toReturn.precision = desiredPrecision;

            return toReturn;
        }

	};


}