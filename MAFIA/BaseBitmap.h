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

///////////////////////////////////////////////////////////////////
///
/// BaseBitmap.h
///
/// A simple bitmap class with only uncompressed data.
/// Note: Name bitmaps are of this type.
///
///////////////////////////////////////////////////////////////////

#ifndef BaseBitmap_H
#define BaseBitmap_H
#include "Tables.h"
#include <stdlib.h>

/// @addtogroup BitmapProcessing Bitmap processing
/// Classes and methods for storing and processing bitmaps
/** @{ */

///////////////////////////////////////////////////////////////////
/// A simple bitmap class with only uncompressed data for name bitmaps
///////////////////////////////////////////////////////////////////
class BaseBitmap {

public:

    /////////////////////////////////////////////////////////////////////
    /// Default constructor
    /////////////////////////////////////////////////////////////////////
    explicit BaseBitmap() {
        _memory = NULL;
    }

    explicit BaseBitmap(int numBits);
    explicit BaseBitmap(BaseBitmap &);

    /////////////////////////////////////////////////////////////////////
    /// Deallocate memory for the BaseBitmap
    /////////////////////////////////////////////////////////////////////
    ~BaseBitmap() {
        if (_memory)
            delete [] _memory;
    }

    void FillRand(double prob);
    void FillOnes();
    int Count(int &CountCounts);

    void Or(const BaseBitmap &b1, const BaseBitmap &b2);
    void AndOnly(const BaseBitmap &b1, const BaseBitmap &b2, int &CountAnds);
    void NotAndOnly(const BaseBitmap &b1, const BaseBitmap &b2, int &CountAnds);
    bool Superset(const BaseBitmap *subset);
    bool SupersetEq(const BaseBitmap *subset);

    friend class Count;
    int _size;                         ///< in number of INTs
    int _count;                        ///< the number of ones in the BaseBitmap
    unsigned int* _memory;             ///< where uncompressed data is stored

    /////////////////////////////////////////////////////////////////////
    /// Check position of bit
    ///
    /// @param CountCheckPosition      # of times this function is called
    /// @param bitIndex          bit position in BaseBitmap to be checked
    /// @return                  0 if bit is 0, 1 otherwise
    /////////////////////////////////////////////////////////////////////
    unsigned int CheckPosition(int bitIndex, int &CountCheckPosition) {
        CountCheckPosition++;
        return (_memory[bitIndex / 32] & BitTable[(bitIndex % 32)]);
    }

    /////////////////////////////////////////////////////////////////////
    /// Check position of bit
    ///
    /// @param bitIndex          bit position in BaseBitmap to be checked
    /// @return                  0 if bit is 0, 1 otherwise
    /////////////////////////////////////////////////////////////////////
    unsigned int CheckPosition(int bitIndex) {
        return (_memory[bitIndex / 32] & BitTable[(bitIndex % 32)]);
    }

    /////////////////////////////////////////////////////////////////////
    /// Fill in a 1 in a certain position
    ///
    /// @param bitIndex          bit position in BaseBitmap to be changed
    /////////////////////////////////////////////////////////////////////
    void FillEmptyPosition(int bitIndex) {
        // Locate correct int in the BaseBitmap
        int byteIndex = bitIndex / 32;

        // switch on correct bit
        _memory[byteIndex] = _memory[byteIndex] | BitTable[(bitIndex % 32)];
    }
};

/** @} */

#endif
