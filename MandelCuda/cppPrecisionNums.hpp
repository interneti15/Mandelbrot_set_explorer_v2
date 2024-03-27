#pragma once
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

namespace myNumLib
{
	constexpr unsigned char UNSIGNED_CHAR_MAX = 255;

    class bigInt 
    {
    public:
        unsigned char* number;// Array representing the number
        size_t SIZE;// Size of the array
        bool sign = true;// true = positive number, false = negative number

        // Constructor
        static bigInt bigIntConstructor(const int MAX_SIZE = 50, bool SIGN = true) {
            bigInt toReturn;

            toReturn.number = new unsigned char[MAX_SIZE]();
            toReturn.SIZE = MAX_SIZE;
            toReturn.sign = SIGN;

            return toReturn;
        }

        // Constructor, when called on kernel thread
        __device__ static bigInt deviceBigIntConstructor(const int MAX_SIZE = 50, bool SIGN = true) {
            bigInt toReturn;

            toReturn.number = new unsigned char[MAX_SIZE]();
            toReturn.SIZE = MAX_SIZE;
            toReturn.sign = SIGN;

            return toReturn;
        }

        static bool isFirstBiggerThenSecond(const bigInt a, const bigInt b) {

            if (a.sign && !b.sign) // if a is positive and b is negative
            {
                return true;
            }

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
            return false;
        }

        __device__ static bool deviceisFirstBiggerThenSecond(const bigInt a, const bigInt b) {

            if (a.sign && !b.sign) // if a is positive and b is negative
            {
                return true;
            }

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
            return false;
        }

        // Extend the lenght of the array storing the data to accomodate higher numbers
        void extendMAX_SIZE(const int desiredSize) {
            
            unsigned char* newNumber = new unsigned char[desiredSize]();

            for (size_t i = 0; i < this->SIZE; i++)
            {
                newNumber[i] = this->number[i];
            }

            delete[] this->number;
            this->number = newNumber;
        }

        // Extend the lenght of the array storing the data to accomodate higher numbers
        __device__ void deviceExtendMAX_SIZE(const int desiredSize) {

            unsigned char* newNumber = new unsigned char[desiredSize]();

            for (size_t i = 0; i < this->SIZE; i++)
            {
                newNumber[i] = this->number[i];
            }

            delete[] this->number;
            this->number = newNumber;
        }

        // Auto trim 0 from the front of the number, used to reduce the allocation size of the object, Manual argument - Trim to be no smaller than this argument, force argument - force manual
        void autoTrim(long long manual = 0, bool force = false) {
            
            unsigned long long countFromLeft = 0;
            long long newSize;

            if (force)
            {
                newSize = manual;

                unsigned char* newNumber = new unsigned char[newSize]();

                for (size_t i = 0; i < newSize; i++)
                {
                    newNumber[i] = this->number[i];
                }
                this->SIZE = newSize;
                delete this->number;
                this->number = newNumber;
            }

            for (long long i = (this->SIZE)-1; i >= 0; i--)
            {
                if (this->number[i] == 0)
                {
                    countFromLeft++;
                }
            }

            if (countFromLeft != 0)
            {
                if (manual != 0)
                {
                    if (this->SIZE - countFromLeft > manual)
                    {
                        newSize = this->SIZE - countFromLeft;
                    }
                    else
                    {
                        newSize = manual;
                    }
                }
                else
                {
                    newSize = this->SIZE - countFromLeft;
                }

                unsigned char* newNumber = new unsigned char[newSize]();

                for (size_t i = 0; i < newSize; i++)
                {
                    newNumber[i] = this->number[i];
                }
                this->SIZE = newSize;
                delete[] this->number;
                this->number = newNumber;
            }
        }

        // Auto trim 0 from the front of the number, used to reduce the allocation size of the object, Manual argument - Trim to be no smaller than this argument, force argument - force manual
        __device__ void deviceAutoTrim(long long manual = 0, bool force = false) {

            unsigned long long countFromLeft = 0;
            long long newSize;

            if (force)
            {
                newSize = manual;

                unsigned char* newNumber = new unsigned char[newSize]();

                for (size_t i = 0; i < newSize; i++)
                {
                    newNumber[i] = this->number[i];
                }
                this->SIZE = newSize;
                delete this->number;
                this->number = newNumber;
            }

            for (long long i = (this->SIZE) - 1; i >= 0; i--)
            {
                if (this->number[i] == 0)
                {
                    countFromLeft++;
                }
            }

            if (countFromLeft != 0)
            {
                if (manual != 0)
                {
                    if (this->SIZE - countFromLeft > manual)
                    {
                        newSize = this->SIZE - countFromLeft;
                    }
                    else
                    {
                        newSize = manual;
                    }
                }
                else
                {
                    newSize = this->SIZE - countFromLeft;
                }

                unsigned char* newNumber = new unsigned char[newSize]();

                for (size_t i = 0; i < newSize; i++)
                {
                    newNumber[i] = this->number[i];
                }
                this->SIZE = newSize;
                delete[] this->number;
                this->number = newNumber;
            }
        }

        static bigInt add(bigInt a, bigInt b) {

            const unsigned long long MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;
            const unsigned long long MIN_SIZE = a.SIZE < b.SIZE ? a.SIZE : b.SIZE;

            bigInt toReturn = bigIntConstructor(MAX_SIZE);

            if (a.sign && b.sign) {
                toReturn.sign = true; // Positive
            }
            else if (!a.sign && !b.sign) {
                toReturn.sign = false; // Negative
            }
            else {
                // One is positive, one is negative, perform subtraction
                if (a.sign) {
                    b.sign = true; // Change the sign of b to positive
                    return subtract(a, b);
                }
                else {
                    a.sign = true; // Change the sign of a to positive
                    return subtract(b, a);
                }
            }

            unsigned char toAddNext = 0;

            for (size_t i = 0; i < MIN_SIZE; i++)
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
            toReturn.number[MIN_SIZE] = toAddNext;

            toAddNext = 0;
            if (MIN_SIZE != MAX_SIZE && a.SIZE > b.SIZE)
            {
                for (size_t i = MIN_SIZE; i < MAX_SIZE; i++)
                {
                    if (a.number[i] + toReturn.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = a.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = a.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }
            }
            else if (MIN_SIZE != MAX_SIZE && a.SIZE < b.SIZE)
            {
                for (size_t i = MIN_SIZE; i < MAX_SIZE; i++)
                {
                    if (b.number[i] + toReturn.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = b.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = b.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }
            }

            if (toAddNext != 0)
            {
                toReturn.extendMAX_SIZE(MAX_SIZE + 1);
            }
            toReturn.number[MAX_SIZE] = toAddNext;

            return toReturn;
        }

        __device__ static bigInt deviceAdd(bigInt a, bigInt b) {

            const int MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;
            const int MIN_SIZE = a.SIZE < b.SIZE ? a.SIZE : b.SIZE;

            bigInt toReturn = deviceBigIntConstructor(MAX_SIZE);
            unsigned char toAddNext = 0;

            for (size_t i = 0; i < MIN_SIZE; i++)
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
            toReturn.number[MIN_SIZE] = toAddNext;

            toAddNext = 0;
            if (MIN_SIZE != MAX_SIZE && a.SIZE > b.SIZE)
            {
                for (size_t i = MIN_SIZE; i < MAX_SIZE; i++)
                {
                    if (a.number[i] + toReturn.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = a.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = a.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }
            }
            else if (MIN_SIZE != MAX_SIZE && a.SIZE < b.SIZE)
            {
                for (size_t i = MIN_SIZE; i < MAX_SIZE; i++)
                {
                    if (b.number[i] + toReturn.number[i] + toAddNext > UNSIGNED_CHAR_MAX)
                    {
                        toReturn.number[i] = b.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 1;
                    }
                    else
                    {
                        toReturn.number[i] = b.number[i] + toReturn.number[i] + toAddNext;
                        toAddNext = 0;
                    }
                }
            }

            if (toAddNext != 0)
            {
                toReturn.deviceExtendMAX_SIZE(MAX_SIZE + 1);
            }
            toReturn.number[MAX_SIZE] = toAddNext;

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


    private:

        // Performs subtraction of two big integers, a and b, with checking for overflow
        static bigInt subtract(const bigInt& a, const bigInt& b) {

            printf("substract\n");

            const unsigned long long MAX_SIZE = a.SIZE > b.SIZE ? a.SIZE : b.SIZE;
            const unsigned long long MIN_SIZE = a.SIZE < b.SIZE ? a.SIZE : b.SIZE;


            // Check if a is greater than or equal to b
            if (isFirstBiggerThenSecond(a, b) || true) {
                // Initialize the difference
                bigInt diff = bigIntConstructor(MAX_SIZE);

                // Perform subtraction using long subtraction algorithm
                bool borrow = false;
                for (int i = 0; i < a.SIZE; ++i) {
                    unsigned long long current = static_cast<unsigned long long>(a.number[i]) - (borrow ? 1 : 0) - static_cast<unsigned long long>(b.number[i] & UNSIGNED_CHAR_MAX);
                    diff.number[i] = static_cast<unsigned char>(current % (UNSIGNED_CHAR_MAX + 1));
                    borrow = (current / (UNSIGNED_CHAR_MAX + 1)) > 0;
                }
                diff.sign = a.sign; // Positive result
                diff.autoTrim(MAX_SIZE);

                return diff;

            }
            else {
                // Change the signs of a and b
                bigInt tempA = a;
                tempA.sign = !tempA.sign;
                bigInt tempB = b;
                tempB.sign = !tempB.sign;

                // Perform addition to get the negative difference
                bigInt diff = add(tempA, tempB);
                diff.sign = false;

                return diff;
            }

            
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