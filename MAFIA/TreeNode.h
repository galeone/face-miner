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
/// TreeNode.h
///
/// A class for representing nodes in the
/// search tree
///
/////////////////////////////////////////////////////////////////////

#ifndef TREENODE_H
#define TREENODE_H

#include "Bitmap.h"
#include "BaseBitmap.h"


/////////////////////////////////////////////////////////////////////
/// @ingroup AlgorithmicComponents
/// Node structure for each node of the search tree
/////////////////////////////////////////////////////////////////////
class TreeNode {
public:

    // Constructors
    TreeNode():
            Name(NULL),
            Trans(NULL),
            Depth(0),            
            tBegin(-1),
            tEnd(0),
            Prefix(0)
        {}
        
    TreeNode(
        BaseBitmap *NAME,
        Bitmap *TRANS,
        int DEPTH,
        int COMP_ID,
        int PREFIX,
        int TBEGIN,
        int TEND):
            Name(NAME),
            Trans(TRANS),
            Depth(DEPTH),
            compID(COMP_ID),
            tBegin(TBEGIN),
            tEnd(TEND),
            Prefix(PREFIX)
        {}

    void setTreeNode(
        BaseBitmap *NAME,
        Bitmap *TRANS,
        int DEPTH,
        int COMP_ID,
        int PREFIX,
        int TBEGIN,
        int TEND)           
        {
            Name = NAME;
            Trans = TRANS;
            Depth = DEPTH;
            compID = COMP_ID;
            tBegin = TBEGIN;
            tEnd = TEND;
            Prefix = PREFIX;
            
        }

    ~TreeNode();

    BaseBitmap *Name;        ///< Bitmap representing the head of the node   
    Bitmap *Trans;           ///< Bitmap storing the list of transactions

    int Depth;               ///< Depth of the node in the search tree
    int compID;              ///< compression operation id
                             
    int tBegin;              ///< where tail begins
    int tEnd;                ///< where tail ends

    int rBegin;              ///< where the relevant itemsets in MFI begin
    int rEnd;                ///< where the relevant itemsets in MFI end
    int Prefix;              ///< last element of head (like a prefix tree)
};


/////////////////////////////////////////////////////////////////////
/// Destructor.  Remove node from memory, but do NOT free up the associated
///     bitmaps, since they can be reused.
/////////////////////////////////////////////////////////////////////
TreeNode::~TreeNode() {
    Trans = NULL;
    Name = NULL;
}

#endif
