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
/// BaseBitmap.cpp
///
/////////////////////////////////////////////////////////////////////

#include "BaseBitmap.h"
#include <limits.h>

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessing
/** @{ */

#define BYTE_COUNT (sizeof(unsigned int) / sizeof(unsigned char))

/////////////////////////////////////////////////////////////////////
/// Allocate the memory for the BaseBitmap
///
/// @param numBits           number of bits in the BaseBitmap
/////////////////////////////////////////////////////////////////////
BaseBitmap::BaseBitmap(int numBits) {
    // Convert bitsize to number of ints
    _size = (numBits / 32) + 1;
    _memory = new unsigned int[_size];
    _count = 0;

    // fill BaseBitmap with zeroes
    for (int index = 0; index < _size; index++)
        _memory[index] = 0;
}

/////////////////////////////////////////////////////////////////////
/// Copy constructor
///
/// @param bitmapToCopy                 BaseBitmap to copy
/////////////////////////////////////////////////////////////////////
BaseBitmap::BaseBitmap(BaseBitmap &bitmapToCopy) {
    // Copy information from source bitmap
    _size = bitmapToCopy._size;
    _memory = new unsigned int[ _size ];
    _count = bitmapToCopy._count;

    for (int byteIndex = 0; byteIndex < bitmapToCopy._size; byteIndex++)
        _memory[byteIndex] = bitmapToCopy._memory[byteIndex];
}

/////////////////////////////////////////////////////////////////////
/// Fill the BaseBitmap with random data
///
/// @param prob                 probability of a bit being set to 1
/////////////////////////////////////////////////////////////////////
void BaseBitmap::FillRand(double prob) {
    unsigned int randy;

    for (int byteIndex = 0; byteIndex < _size; byteIndex++) {
        randy = 0;
        for (int bitIndex = 0; bitIndex < 32; bitIndex++) {
            // Fill bit with 1 when below probability prob
            if ((rand() / (float)RAND_MAX) < prob)
                randy = randy + BitTable[bitIndex];
        }

        _memory[byteIndex] = randy;
    }
}

/////////////////////////////////////////////////////////////////////
/// Fill the BaseBitmap with ones
/////////////////////////////////////////////////////////////////////
void BaseBitmap::FillOnes() {
    for (int byteIndex = 0; byteIndex < _size; byteIndex++)
        _memory[byteIndex] = UINT_MAX;
}

/////////////////////////////////////////////////////////////////////
/// Count the ones in the bitmap
///
/// @param CountCounts       a counter of counts (for debugging)
/// @return                  the count of ones
/////////////////////////////////////////////////////////////////////
int BaseBitmap::Count(int &CountCounts) {
    int final = 0;
    unsigned char *currentIndex;
    for (int byteIndex = 0; byteIndex < _size; byteIndex++) {
        currentIndex = (unsigned char *) & _memory[byteIndex];

        // Count the 4 chars (1 int = 4 bytes/chars)
        // count 1s by dividing into chars and using a lookup table
        final += CountTable[*currentIndex];
        currentIndex++;
        final += CountTable[*currentIndex];
        currentIndex++;
        final += CountTable[*currentIndex];
        currentIndex++;
        final += CountTable[*currentIndex];
    }

    CountCounts++;

    // set the count for the BaseBitmap
    _count = final;
    return final;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise OR 2 bitmaps and store the result
///
/// @param B1                the first Bitmap
/// @param B2                the second Bitmap
/////////////////////////////////////////////////////////////////////
void BaseBitmap::Or(const BaseBitmap &B1, const BaseBitmap &B2) {
    unsigned int *b1ptr, *b2ptr;
    b1ptr = B1._memory;
    b2ptr = B2._memory;

    // for each INT in the bitmaps
    for (int byteIndex = 0; byteIndex < B1._size; byteIndex++) {
        /// Bitwise OR the int
        _memory[byteIndex] = (*b1ptr) | (*b2ptr);
        b1ptr++;
        b2ptr++;
    }

    // Update the count (B1 and B2 are mutually exclusive)
    _count = B1._count + B2._count;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 bitmaps and store the result
///
/// @param B1                the first Bitmap
/// @param B2                the second Bitmap
/// @param CountAnds         [output] counter for # of ANDs (for debugging)
/////////////////////////////////////////////////////////////////////
void BaseBitmap::AndOnly(
    const BaseBitmap &B1,
    const BaseBitmap &B2,
    int &CountAnds) {

    CountAnds++;
    unsigned int *b1ptr, *b2ptr;

    // AND the bitmaps

    b1ptr = B1._memory;
    b2ptr = B2._memory;

    for (int byteIndex = 0; byteIndex < B1._size; byteIndex++) {
        // and each int bitwise
        _memory[byteIndex] = (*b1ptr) & (*b2ptr);
        b1ptr++;
        b2ptr++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 bitmaps and the negation and store the result
///
/// @param B1                the first Bitmap
/// @param B2                the second Bitmap
/// @param CountAnds         [output] counter for # of ANDs (for debugging)
/////////////////////////////////////////////////////////////////////
void BaseBitmap::NotAndOnly(
    const BaseBitmap &B1,
    const BaseBitmap &B2,
    int &CountAnds) {

    CountAnds++;
    unsigned int *b1ptr, *b2ptr;

    // AND the small bitmaps

    b1ptr = B1._memory;
    b2ptr = B2._memory;

    for (int byteIndex = 0; byteIndex < B1._size; byteIndex++) {
        // and each int bitwise
        _memory[byteIndex] = (*b1ptr) & ~(*b2ptr);
        b1ptr++;
        b2ptr++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Determine whether this bitmap is a superset of the parameter.
/// NOTE:  Assumes the set of bitmaps is lexicographically ordered.
///
/// @param subset            bitmap to check for a superset relation
/// @return                  True if this bitmap is a superset of the
///                          parameter bitmap (this bitmap has a 1 in ALL
///                          positions that parameter bitmap does )
/////////////////////////////////////////////////////////////////////
bool BaseBitmap::Superset(const BaseBitmap *subset) {
    // No need to check for superset when the count is too small
    if (_count <= subset->_count) {
        return false;
    }

    // Start at the end of the bitmap
    // Leads to fewer bitwise ANDs
    unsigned int *b1ptr, *b2ptr;
    b1ptr = &_memory[_size - 1];
    b2ptr = &subset->_memory[subset->_size - 1];

    // for each INT in the bitmaps
    for (int byteIndex = _size - 1; byteIndex >= 0; byteIndex--) {
        // Andy++;
        // bitwise AND them
        unsigned int andResult = (*b1ptr) & (*b2ptr);

        // if not a subset
        if (andResult != *b2ptr)
            return false;

        b1ptr--;
        b2ptr--;
    }

    return true;
}

/////////////////////////////////////////////////////////////////////
/// Determine whether this bitmap is a superset of the parameter or equal to it
///
/// @param subset            bitmap to check for a superseteq relation
/// @return                  True if this bitmap is a superset of the
///                          parameter bitmap (this bitmap has a 1 in ALL
///                          positions that parameter bitmap does )
/////////////////////////////////////////////////////////////////////
bool BaseBitmap::SupersetEq(const BaseBitmap *subset) {
    // No need to check for superset when the count is too small
    if (_count < subset->_count) {
        return false;
    }

    // Start at the end of the bitmap
    // Leads to fewer bitwise ANDs
    unsigned int *b1ptr, *b2ptr;
    b1ptr = &_memory[_size - 1];
    b2ptr = &subset->_memory[subset->_size - 1];

    // for each INT in the bitmaps
    for (int byteIndex = _size - 1; byteIndex >= 0; byteIndex--) {
        // Andy++;
        // bitwise AND them
        unsigned int andResult = (*b1ptr) & (*b2ptr);

        // if not a subset
        if (andResult != *b2ptr)
            return false;

        b1ptr--;
        b2ptr--;
    }

    return true;
}

/** @} */
