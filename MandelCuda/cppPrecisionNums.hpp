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

        void extendMAX_SIZE(const int MAX_SIZE) {
            
            unsigned char* newNumber = new unsigned char[MAX_SIZE]();

            for (size_t i = 0; i < this->SIZE; i++)
            {
                newNumber[i] = this->number[i];
            }

            delete this->number;
            this->number = newNumber;
        }

        __device__ void deviceExtendMAX_SIZE(const int MAX_SIZE) {

            unsigned char* newNumber = new unsigned char[MAX_SIZE]();

            for (size_t i = 0; i < this->SIZE; i++)
            {
                newNumber[i] = this->number[i];
            }

            delete this->number;
            this->number = newNumber;
        }

        static bigInt add(bigInt a, bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;

            if (a.SIZE < b.SIZE)
            {
                a.extendMAX_SIZE(b.SIZE);
            }
            else if (b.SIZE < a.SIZE)
            {
                b.extendMAX_SIZE(a.SIZE);
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

            if (a.SIZE < b.SIZE)
            {
                a.deviceExtendMAX_SIZE(b.SIZE);
            }
            else if (b.SIZE < a.SIZE)
            {
                b.deviceExtendMAX_SIZE(a.SIZE);
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

        // Long multiplication
        static bigInt multiply(const bigInt& a, const bigInt& b) { 
            // Determine the size of the result array
            const int resultSize = a.SIZE + b.SIZE;

            // Initialize the result to store the multiplication result
            bigInt result = bigIntConstructor(resultSize);

            // Perform multiplication using long multiplication algorithm
            for (int i = 0; i < a.SIZE; ++i) {
                unsigned char carry = 0;
                for (int j = 0; j < b.SIZE || carry; ++j) {
                    unsigned long long current = result.number[i + j] + (unsigned long long)a.number[i] * (j < b.SIZE ? b.number[j] : 0) + carry;
                    result.number[i + j] = static_cast<unsigned char>(current % (UNSIGNED_CHAR_MAX + 1));
                    carry = static_cast<unsigned char>(current / (UNSIGNED_CHAR_MAX + 1));
                }
            }
            /*
            // Trim leading zeros if any
            int newSize = resultSize;
            while (newSize > 1 && result.number[newSize - 1] == 0) {
                newSize--;
            }

            // Update the result size
            result.SIZE = newSize;
            */
            return result;
        }

        // Long multiplication for on kernel usage
        __device__ static bigInt deviceMultiply(const bigInt& a, const bigInt& b) {
            // Determine the size of the result array
            const int resultSize = a.SIZE + b.SIZE;

            // Initialize the result to store the multiplication result
            bigInt result = deviceBigIntConstructor(resultSize);

            // Perform multiplication using long multiplication algorithm
            for (int i = 0; i < a.SIZE; ++i) {
                unsigned char carry = 0;
                for (int j = 0; j < b.SIZE || carry; ++j) {
                    unsigned long long current = result.number[i + j] + (unsigned long long)a.number[i] * (j < b.SIZE ? b.number[j] : 0) + carry;
                    result.number[i + j] = static_cast<unsigned char>(current % (UNSIGNED_CHAR_MAX + 1));
                    carry = static_cast<unsigned char>(current / (UNSIGNED_CHAR_MAX + 1));
                }
            }

            /*
            // Trim leading zeros if any
            int newSize = resultSize;
            while (newSize > 1 && result.number[newSize - 1] == 0) {
                newSize--;
            }

            // Update the result size
            result.SIZE = newSize;
            */
            return result;
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