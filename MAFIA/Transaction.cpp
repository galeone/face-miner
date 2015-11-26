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
/// Transaction.cpp
///
/////////////////////////////////////////////////////////////////////

#include "Transaction.h"
#include <vector>
#include <algorithm>
using namespace std;

/////////////////////////////////////////////////////////////////////
/// @addtogroup InputOutput
/** @{ */

/////////////////////////////////////////////////////////////////////
/// Get next transaction from the input file
///
/// @return     pointer to a new Transaction
/////////////////////////////////////////////////////////////////////
Transaction *InputData::getNextTransaction() {
    char c;

    int itemIndex = 0;
    if (isAsciiFile) {
        // read list of items
        do {
            int item = 0, pos = 0;
            inputFile.get(c);
            while (!inputFile.eof() && (c >= '0') && (c <= '9')) {
                item *= 10;
                item += int(c) - int('0');
                inputFile.get(c);
                pos++;
            }

            if (pos) {
                itembuffer[itemIndex] = item;
                itemIndex++;
            }
        } while (!inputFile.eof() && c != '\n');

        // if end of file is reached
        if (itemIndex == 0)
            return 0;
    } else {
        int custid;              // customer id (NOT used currently)
        int transid;             // transaction id
        int nitems = 0;              // number of items in the transaction

        if (!inputFile.eof()) {
            // read in the transaction
            inputFile.read((char *)&custid, sizeof(int));
            inputFile.read((char *)&transid, sizeof(int));
            inputFile.read((char *)&nitems, sizeof(int));      

            // ensure that there are not too many items
            if (nitems >= MAX_NUM_ITEMS) {
                cout << "More than " << MAX_NUM_ITEMS
                << " items in customer id: " << custid
                << " transaction id: " << transid;
                exit(1);
            }

            // Read in the items of the transaction
            inputFile.read((char *)itembuffer, nitems * sizeof(int));
        }
        
        itemIndex = nitems;
        if (itemIndex == 0)
            return 0;
    }

    // Note, also last transaction must end with newline,
    // else, it will be ignored

    // sort list of items (this is not necessary for the workshop test datasets)
    // sort(list.begin(),list.end());

    // put items in Transaction structure
    Transaction *newTransaction = new Transaction(itembuffer, itemIndex);
    return newTransaction;
}

/////////////////////////////////////////////////////////////////////
/// Open the input file
///
/// @param filename         input filename
/// @param ITEMBUFFER       pointer to buffer for storing a transaction
/// @param IS_ASCII         true if file is in ASCII format
/////////////////////////////////////////////////////////////////////
InputData::InputData(char *filename, int *ITEMBUFFER, bool IS_ASCII) {
    if (IS_ASCII)
        inputFile.open(filename);
    else
        inputFile.open(filename, ios::binary);

    isAsciiFile = IS_ASCII;
    itembuffer = ITEMBUFFER;
}

/////////////////////////////////////////////////////////////////////
/// Close the input file
/////////////////////////////////////////////////////////////////////
InputData::~InputData() {
    inputFile.close();
}

/////////////////////////////////////////////////////////////////////
/// Check if the input file is open
///
/// @return true if file is open
/////////////////////////////////////////////////////////////////////
int InputData::isOpen() {
    return inputFile.is_open();
}
/** @} */
