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
/// Transaction.h
///
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

using namespace std;

#define MAX_NUM_ITEMS 1000000
#define MAX_ITEMSET_SIZE 10000

/////////////////////////////////////////////////////////////////////
/// @addtogroup InputOutput
/** @{ */

/////////////////////////////////////////////////////////////////////
/// Transaction (itemlist) from a database
/////////////////////////////////////////////////////////////////////
class Transaction {
public:

    Transaction(int *ITEMLIST, int LENGTH) : length(LENGTH), itemlist(ITEMLIST)
    {}
        
    ~Transaction() {}

    int length;             ///< length of the transaction
    int *itemlist;          ///< list of items
};

/////////////////////////////////////////////////////////////////////
/// Class for reading transactions from an ASCII file
/////////////////////////////////////////////////////////////////////
class InputData {
public:

    InputData(char *filename, int *ITEMBUFFER, bool IS_ASCII);
    ~InputData();
    int isOpen();

    Transaction *getNextTransaction();

    int *itembuffer;
    bool isAsciiFile;

private:

    ifstream inputFile;              ///< pointer to file input
};
/** @} */
