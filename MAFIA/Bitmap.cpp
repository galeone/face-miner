/*

Copyright (c) 2003, Cornell University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    - Neither the name of Cornell University nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

*/


/////////////////////////////////////////////////////////////////////
///
/// Bitmap.cpp
///
/////////////////////////////////////////////////////////////////////

#include "Bitmap.h"
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <iostream>
#include <assert.h>

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessing
/** @{ */

#define BYTE_COUNT (sizeof(unsigned int) / sizeof(unsigned char))

/////////////////////////////////////////////////////////////////////
/// Allocate the memory for the bitmap
///
/// @param numBits           number of bits in the Bitmap
/////////////////////////////////////////////////////////////////////
Bitmap::Bitmap(int numBits) {
    // Convert bitsize to number of ints
    _size = (numBits / 32) + 1;
    _compSize = (int)(_size * 1.25) + 1;
    _memory = new unsigned int[ _size ];
    _compMemory = new unsigned int[_compSize ];
    _compUsed = 0;
    _count = 0;

    int index;

    // fill bitmap with zeroes
    for (index = 0; index < _size; index++)
        _memory[index] = 0;

    // fill compressed bitmap with zeroes
    for (index = 0; index < _compSize; index++)
        _compMemory[index] = 0;
}

/////////////////////////////////////////////////////////////////////
/// Copy constructor
///
/// @param b                 bitmap to copy
/////////////////////////////////////////////////////////////////////
Bitmap::Bitmap(Bitmap &b): BaseBitmap(b) {
    _size = b._size;
    _compSize = b._compSize;
    _compUsed = b._compUsed;
    _memory = new unsigned int[ _size ];
    _compMemory = new unsigned int[ _compSize ];
    _count = b._count;

    int index;
    for ( index = 0; index < b._size; index++)
        _memory[index] = b._memory[index];

    for ( index = 0; index < b._compSize; index++)
        _compMemory[index] = b._compMemory[index];
}


/////////////////////////////////////////////////////////////////////
/// Deallocate memory for the Bitmap
/////////////////////////////////////////////////////////////////////
Bitmap::~Bitmap() {
    delete [] _compMemory;
}

/////////////////////////////////////////////////////////////////////
/// Fill in a 1 in a certain position in COMPRESSED data
///
/// @param j                 bit position in Bitmap to be changed
/////////////////////////////////////////////////////////////////////
void Bitmap::FillCompEmptyPosition(int j) {
    // Locate correct int in the bitmap
    int i = j / 32;

    // switch on correct bit
    _compMemory[i] = _compMemory[i] | BitTable[(j % 32)];
}

/////////////////////////////////////////////////////////////////////
/// Count the number of ones in the Compressed bitmap
///
/// @return the count of ones in the bitmap
/////////////////////////////////////////////////////////////////////
int Bitmap::SmallCount(int &CountCounts) {
    int final = 0;
    unsigned char *p;
    CountCounts++;

    for (int i = 0; i < _compUsed; i++) {
        p = (unsigned char *) & _compMemory[i];

        // count 1s by dividing into chars and using a lookup table
        final += CountTable[*p];
        p++;
        final += CountTable[*p];
        p++;
        final += CountTable[*p];
        p++;
        final += CountTable[*p];
    }

    // set the count for the bitmap
    _count = final;
    return final;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 compressed bitmaps and store the result
///
/// @param B1                first bitmap
/// @param B2                second bitmap
/// @param CountSmallAnds    [output] counter of ANDs on compressed data
/////////////////////////////////////////////////////////////////////
void Bitmap::AndCompOnly(
            const Bitmap &B1,
            const Bitmap &B2,
            int &CountSmallAnds) {
            
    CountSmallAnds++;
    unsigned int *b1ptr, *b2ptr;

    b1ptr = B1._compMemory;
    b2ptr = B2._compMemory;
    _compUsed = B1._compUsed;

    // AND the small bitmaps
    for (int i = 0; i < B1._compUsed; i++) {
        // and each INT bitwise
        _compMemory[i] = (*b1ptr) & (*b2ptr);

        b1ptr++;
        b2ptr++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 compressed bitmaps and the negation and store the result
///
/// @param B1                first bitmap
/// @param B2                second bitmap
/// @param CountSmallAnds    [output] counter of ANDs on compressed data
/////////////////////////////////////////////////////////////////////
void Bitmap::NotAndCompOnly(
            const Bitmap &B1,
            const Bitmap &B2,
            int &CountSmallAnds) {
            
    CountSmallAnds++;
    unsigned int *b1ptr, *b2ptr;

    b1ptr = B1._compMemory;
    b2ptr = B2._compMemory;
    _compUsed = B1._compUsed;

    // AND the small bitmaps
    for (int i = 0; i < B1._compUsed; i++) {
        // and each INT bitwise
        _compMemory[i] = (*b1ptr) & ~(*b2ptr);

        b1ptr++;
        b2ptr++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Compress this bitmap relative to the source - has a bit for each
///    trans in source
///
/// @param source            the bitmap you are compressing relative to
/////////////////////////////////////////////////////////////////////
void Bitmap::BuildRelComp(Bitmap &source) {
    int i;
    int bc;

    // index into current int (how much of current int has been filled)
    int d = 0;  

    // zero your compressed contents
    _compUsed = source._compUsed;
    for (i = 0; i < _compUsed; i++)
        _compMemory[i] = 0;

    unsigned int *result = _compMemory;
    unsigned int tempResult;
    unsigned char *p_s, *p_t;

    p_s = (unsigned char *) source._memory;
    p_t = (unsigned char *) _memory;

    // for each byte of the ORIGINAL bitmaps
    for (i = 0; i < (_size * 4); i++) {
        // find out how many bits needed to represent this byte
        //     in compressed form
        // (# of 1's in the ORIGINAL data of the SOURCE)
        bc = CountTable[(*p_s)];

        // if there is anything when byte compressed
        if (bc > 0) {
            // check if enough space in current int of COMPRESSED data
            //     for the compressed byte result
            if ((32 - d) >= bc) {
                // compress and append compressed result for this byte to
                //     rest of compressed data
                tempResult = CompTable[(*p_s)][(*p_t)] >> d;
                *result = (*result) | tempResult;
                d += bc;
            } else {
                // go to the next int in compressed data, compress and
                //     append compressed result there
                result++;
                tempResult = CompTable[(*p_s)][(*p_t)];
                *result = (*result) | tempResult;
                d = bc;
            }

        }
        
        // go to the byte in the ORIGINAL bitmaps
        p_s++;
        p_t++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Fill the compressed bitmap with ones
/////////////////////////////////////////////////////////////////////
void Bitmap::BuildSource() {
    unsigned int *whatever = _compMemory;

    // find out how many INTs will be used in the compressed data
    int scaledSup = (int)ceil( ((float)_count * 1.25) / (float)32 );
    for (int i = 0; i < scaledSup; i++) {
        // fill the int with all ones
        (*whatever) = UINT_MAX;
        whatever++;
    }

    _compUsed = scaledSup;
}

/** @} */
