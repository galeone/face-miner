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
//
// Mafia.cpp
//
/////////////////////////////////////////////////////////////////////


#include "Mafia.h"

using namespace std;
using namespace MAFIA;

/// @defgroup GlobalVariables Global Variables
/// Global variables
/** @{ */


/// @defgroup CommandLineParameters Command Line Parameters/Variables
/// Commmand line parameters from user or inferred
/** @{ */
string method;               ///< either -mfi or -fci or -fi
char* outFilename;           ///< filename for output
ItemsetOutput *outFile;      ///< file for ouput
bool outputMFI = false;      ///< true if MFI should be saved to a file
bool MethodIsFI = false;     ///< true if the method is -fi
bool MethodIsFCI = false;    ///< true if the method is -fci
int ItemCount;               ///< # of items in the file
int TransCount;              ///< # of transactions in the file
double MSF;                  ///< user-defined min sup as percentage
int MS;                      ///< min sup as a transaction count
int VerScale = 1;            ///< Scaling factor for transactions
int HorScale = 1;            ///< Scaling factor for items
bool GoFHUT = true;          ///< FHUT flag
bool HUTMFI = true;          ///< HUTMFI flag
bool PEPrune = true;         ///< PEPrune flag -- parent equivalent pruning
bool Reorder = true;         ///< Reorder flag
/** @} */


/// @defgroup CounterVariables Counter Variables
/// Counter variables for information gathering
/** @{ */
int CountFHUT = 0;           ///< # of times FHUT was successful
int CountNodes = 0;          ///< # of frequent nodes in the tree
int CountCounts = 0;         ///< # of Counts or all nodes in the tree
int CountAnds = 0;           ///< # of ANDs of normal bitmaps
int CountSmallAnds = 0;      ///< # of compressed bitmap ANDs
int CountPEPrunes = 0;         ///< # of PEPruning
int CountCheckPosition = 0;  ///< # of CheckPosition calls
int CountHUTMFI = 0;         ///< # of HUTMFI attempts
int CountHUTMFISuccess = 0;  ///< # of HUTMFI successes
int CountRebuilds;           ///< # of Rebuilds
/** @} */


/// @defgroup ProgramVariables Program Parameters/Variables
/// Useful program parameters/counters
/** @{ */
int maxtail = 0;
int MFISize = 0;             ///< MFI size before pruning
int MFIDepth = 0;            ///< The aggregated depth of the all MFI elements
int F1size = 0;              ///< # of frequent 1-itemsets after merging repeats
int FullF1size = 0;          ///< # of frequent 1-itemsets
int k = 50;                  ///< # of items checked for a MFI lookup
int MAX_compID = 1;          ///< max compression ID
int projectDepth = -1;       ///< depth of the bitmap you're projecting from
int EstimateSize;            ///< size of subtree estimation buffer
int EstimateDiv = 5;         ///< bucket size by frequent tail length
int maxItemsetSize = 0;      ///< max size of a frequent itemset
/** @} */


/// @defgroup DataVariables Data variables
/// Complex data structure variables
/** @{ */
Mafia::NodeList F1;                 ///< List of frequent 1-itemsets
Mafia::BitmapList TransBuffy;       ///< Buffer of transaction bitmaps
Mafia::BaseBitmapList NameBuffy;    ///< Buffer of name bitmaps
Mafia::NodeList NodeBuffy;          ///< Buffer of tree nodess
TreeNode *Root;              ///< The root (the nullset)
Mafia::TailElement *gTail;          ///< global tail pointer
Mafia::TailElement *TailBuffy;      ///< Common Buffer for tail elements
Bitmap *NullTrans;           ///< A transaction bitmap filled with ones
int *ItemMap;                ///< For renaming items after sorting by support
int *ReverseItemMap;         ///< For remembering the renaming...
Mafia::BaseBitmapList MFI;          ///< List of Maximally Frequent Itemsets
Mafia::HashTable HT;                ///< Hash table of transaction supports
vector <int> SupportCountList;       ///< List that stores support count
BaseBitmap* TempName;        ///< Temporary buffer for one name bitmap
Mafia::SubtreeEstimate* EstimateBuffy;      ///< Buffer of subtree estimates
int *MFIBySizes;             ///< Buffer for counting MFI by itemset size
int *ItemsetBuffy;           ///< Buffer for writing itemsets to file output
/** @} */


/// @defgroup TimingVariables Timing Variables
/// Variables for timing (and instrumenting the code)
/** @{ */
time_t total_start, total_finish;
double total_time;
time_t read_start, read_finish;
double read_time;
time_t algorithm_start, algorithm_finish;
double algorithm_time;
time_t print_start, print_finish;
double print_time;
/** @} */
/** @} */

/*********************************************************************
 Reading the data in and building the item bitmaps
*********************************************************************/
/// @addtogroup InputOutput Input/Output Functions
/// Reading the data in and building the item bitmaps.
/// Outputting the frequent itemsets.
/** @{ */

/////////////////////////////////////////////////////////////////////
/// Insert pointer to node in order of increasing support
///
/// @param newNode           item node to add to F1
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::AddToF1(TreeNode *newNode) {
    if (F1.empty())
        F1.push_back(newNode);
    else {
        // use insertion sort for increasing support ordering
        NodeList::iterator noli = F1.begin();
        while (noli != F1.end()) {
            if ((*noli)->Trans->_count >= newNode->Trans->_count) {
                F1.insert(noli, newNode);
                return ;
            }
            noli++;
        }

        // Add to end of F1 list
        F1.push_back(newNode);
    }
}

/////////////////////////////////////////////////////////////////////
/// Create bitmaps filled randomly with probability p
///
/// @param P - probability of a bit p being set
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::F1UsingProb(double P) {
    int blah = 0;
    //cout << "Creating bitmap with prob p=" << P << endl;

    // Create frequent 1-itemsets with probability p
    for (int i = 0; i < ItemCount / HorScale; i++) {
        // Create random transaction bitmap
        Bitmap *trans = new Bitmap(TransCount);
        trans->FillRand(P);

        // Add frequent items to list
        if (trans->Count(blah) >= MS) {
            TreeNode *node = new TreeNode(NULL, trans, 1, -1, i, -1, 0);
            for (int r = 0; r < HorScale; r++)
                AddToF1(node);
        } else
            delete trans;
    }
}

/////////////////////////////////////////////////////////////////////
/// Read transaction data from file and build item bitmaps.
///     If the data is in ascii form:
///     [itemid_1] [itemid_2] ... [itemid_n]
///     If the data is in binary form:
///     [custid] [transid] [number of items] [itemid_1] [itemid_2] ... [itemid_n]
///
/// @param filename          name of file to be read in
/// @param isAsciiFile       true for ASCII input, false for binary
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::F1FromFile(char *filename, bool isAsciiFile) {
    TransCount = 0;
    int MAXitemID = -1;
    int *Counters = new int[MAX_NUM_ITEMS];        //to count the support of items
    int *F1IndexList = new int[MAX_NUM_ITEMS];     // the id's of F1 items
    int *InvF1IndexList = new int[MAX_NUM_ITEMS];
    int itemIndex = 0;

    //Initialize counters
    for (int ct = 0; ct < MAX_NUM_ITEMS; ++ct) {
        Counters[ct] = 0;
        F1IndexList[ct] = -1;
        InvF1IndexList[ct] = -1;
    }

    time(&read_start);
    BitmapList Trans;

    int *itemlist = new int[MAX_NUM_ITEMS];
    InputData *inData = new InputData(filename, itemlist, isAsciiFile);
    if (!inData->isOpen()) {
        cerr << "Input file not found!" << endl;
        exit(1);
    }

    Transaction *newTransaction = inData->getNextTransaction();
    while (newTransaction != NULL) {
        for (itemIndex = 0; itemIndex < newTransaction->length; itemIndex++) {
            // ensure that there are not too many items
            if (newTransaction->itemlist[itemIndex] >= MAX_NUM_ITEMS) {
                cerr << "Read item_id=" << newTransaction->itemlist[itemIndex]
                << " which is more than the max item_id allowed (" << MAX_NUM_ITEMS << ")";
                exit(1);
            }

            if (newTransaction->itemlist[itemIndex] > MAXitemID)
                MAXitemID = newTransaction->itemlist[itemIndex];

            Counters[newTransaction->itemlist[itemIndex]]++;
        }

        TransCount++;
        delete newTransaction;
        newTransaction = inData->getNextTransaction();
    }

    delete newTransaction;
    delete inData;

    MS = (int)ceil(MSF * (double)TransCount);
    //MSF = MS/(double)TransCount;

    ItemCount = MAXitemID + 1;

    int F1items = 0;
    // build the normal bitmaps -- Preallocated memory for the bitmaps
    for (itemIndex = 0; itemIndex <= MAXitemID; itemIndex++) {
        if (Counters[itemIndex] >= MS) {
            F1IndexList[F1items++] = itemIndex;
            InvF1IndexList[itemIndex] = F1items - 1;
            Bitmap *trans = new Bitmap(TransCount);
            trans->_count = Counters[itemIndex];
            Trans.push_back(trans);
        }
    }

    int transIndex = 0;
    InputData *inData2 = new InputData(filename, itemlist, isAsciiFile);
    newTransaction = inData2->getNextTransaction();
    while (newTransaction != NULL) {
        for (int itemIndex = 0; itemIndex < newTransaction->length; itemIndex++) {
            if (InvF1IndexList[newTransaction->itemlist[itemIndex]] != -1) {
                Trans[InvF1IndexList[newTransaction->itemlist[itemIndex]]]->FillEmptyPosition(transIndex);
            }
        }

        transIndex++;
        delete newTransaction;
        newTransaction = inData2->getNextTransaction();
    }

    delete newTransaction;
    delete inData2;
    delete [] itemlist;

    time(&read_finish);
    read_time = difftime(read_finish, read_start);
    printf("Reading input time:    %.2f seconds.\n", read_time);

    // Create F1
    BitmapList::iterator bli = Trans.begin();
    itemIndex = 0;

    while (bli != Trans.end()) {
        TreeNode *node = new TreeNode(
                             NULL,
                             (*bli),
                             1,
                             -1,
                             F1IndexList[itemIndex],
                             -1,
                             0);
        AddToF1(node);
        bli++;
        itemIndex++;
    }

    delete [] Counters;
    delete [] F1IndexList;
    delete [] InvF1IndexList;
}

/////////////////////////////////////////////////////////////////////
/// To print the MFI data out to an ASCII file
///     with each entry having the format:
///     [list of items in MFI entry...] [(support)]
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::PrintMFI() {
    // open output file
    outFile = new ItemsetOutput(outFilename);
    if (!outFile->isOpen()) {
        cerr << "Output file not open!" << endl;
        exit(1);
    }

    if (FullF1size != 0) {
        // translate bitmap to list of INTs
        int* ITEMS = new int[FullF1size];
        for (int i = 0; i < MFISize; i++) {
            int j = 0;
            for (int cc = 0; cc < FullF1size; cc++) {
                if (MFI[i]->CheckPosition(cc, CountCheckPosition) > 0) {
                    ITEMS[j] = ItemMap[cc];
                    j++;
                }
            }

            outFile->printSet(MFI[i]->_count, ITEMS, SupportCountList[i]);
        }
        delete [] ITEMS;
    } else {
        outFile->printSet(0, NULL, TransCount);
    }

    delete outFile;
}

/** @} */


/*********************************************************************
 Algorithmic components
*********************************************************************/
/// @defgroup AlgorithmicComponents Algorithmic Component Functions
/// Algorithm components (HUTMFI, PEP, etc.)
/** @{ */

/////////////////////////////////////////////////////////////////////
/// Check for an existing superset of name in the MFI
///
/// @param location          The node we're at.  Use this to examine only
///                          the relevant portion of the MFI
/// @return True - if superset found.
///         False - if no superset.
/////////////////////////////////////////////////////////////////////
bool MAFIA::Mafia::LMFISuperSet(TreeNode* location) {
    return (location->rEnd > location->rBegin);
}

/////////////////////////////////////////////////////////////////////
/// Output itemset (don't need to save bitmaps for FI)
///
/// @param C                 the current node
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::AddToFI(TreeNode *C) {
    int itemsetIndex = 0;
    for (int cc = 0; cc < FullF1size; cc++) {
        if (C->Name->CheckPosition(cc, CountCheckPosition) > 0) {
            ItemsetBuffy[itemsetIndex] = ItemMap[cc];
            itemsetIndex++;
        }
    }

    MFIBySizes[C->Name->_count]++;
    if (C->Name->_count > maxItemsetSize)
        maxItemsetSize = C->Name->_count;

    if (outputMFI)
        outFile->printSet(C->Name->_count, ItemsetBuffy, C->Trans->_count);

    // Update stat variables
    MFIDepth += C->Name->_count;
    MFISize++;
}

/////////////////////////////////////////////////////////////////////
/// Add this node's name bitmap to the MFI list
///
/// @param C                 the current node
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::AddToMFI(TreeNode *C) {
    // copy name bitmap
    BaseBitmap *name = new BaseBitmap(*C->Name);

    // add to MFI
    MFI.push_back(name);

    // update the end of the relevant
    C->rEnd++;

    // Update stat variables
    MFIDepth += C->Name->_count;
    MFISize++;

    SupportCountList.push_back(C->Trans->_count);
    MFIBySizes[name->_count]++;
    if (name->_count > maxItemsetSize)
        maxItemsetSize = name->_count;
}

/////////////////////////////////////////////////////////////////////
/// Add this node's name bitmap to the FCI list
///
/// @param C                 the current node
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::AddToFCI(TreeNode *C) {
    // Search HT
    HashTable::iterator h = HT.find(C->Trans->_count);

    // If the support of node C is NOT in HashSup
    if (h == HT.end()) {
        // Add a new list to the HashSup
        ItemSet* newList = new ItemSet();
        newList->reserve(500);
        newList->push_back(MFI.size());
        HT.insert(HashTable::value_type(C->Trans->_count, newList));

        // Else add pointer to last item in iName to HT entry
    } else {
        for (ItemSet::reverse_iterator goli = (*h).second->rbegin();
                goli != (*h).second->rend();
                goli++)
            if (MFI[*goli]->Superset(C->Name))
                return ;

        // search the table
        (*h).second->push_back(MFI.size());
    }

    // copy name bitmap
    BaseBitmap *name = new BaseBitmap(*C->Name);

    // add to MFI
    MFI.push_back(name);

    // Update stat variables
    MFIDepth += C->Name->_count;
    MFISize++;

    SupportCountList.push_back(C->Trans->_count);
    MFIBySizes[name->_count]++;
    if (name->_count > maxItemsetSize)
        maxItemsetSize = name->_count;
}

int MAFIA::Mafia::SortLMFI(int rBegin, int rEnd, int sortBy) {
    int left = rBegin;
    int right = rEnd - 1;
    while (left <= right) {
        while (left < rEnd && !MFI[left]->CheckPosition(sortBy, CountCheckPosition))
            left++;
        while (right >= rBegin && MFI[right]->CheckPosition(sortBy, CountCheckPosition))
            right--;
        if (left < right) {
            // we are now at a point where MFI[left] is relevant
            // and MFI[right] is not since left < right, we swap the two
            BaseBitmap* tempBitmap = MFI[left];
            MFI[left] = MFI[right];
            MFI[right] = tempBitmap;
            tempBitmap = NULL;

             int tempSupport = SupportCountList[left];
             SupportCountList[left] = SupportCountList[right];
             SupportCountList[right] = tempSupport;
            
            left++;
            right--;
        }
    }

    // the first relevant one for the next node is left
    return left;
}

/////////////////////////////////////////////////////////////////////
/// Determine whether a HUTMFI is true.
///     - if HUT is in MFI, then HUT is frequent
///     and the subtree rooted at this node can be pruned
///
/// @return True if HUT is in the MFI
/////////////////////////////////////////////////////////////////////
bool MAFIA::Mafia::CheckHUTMFI(TreeNode *C, int iTAIL) {

    // for each element i in the tail form {head} U {i} and check for a
    //     superset in the MFI
    int rBegin = C->rBegin;
    int rEnd = C->rEnd;
    for (; iTAIL < C->tEnd; iTAIL++) {
        rBegin = SortLMFI(rBegin, rEnd, F1[gTail[iTAIL].Item]->Prefix);

        if (rEnd <= rBegin)
            return false;
    }

    return true;
}

/////////////////////////////////////////////////////////////////////
/// Dynamically reorder the elements in the tail by increasing support
///    - Expand all children and sort by increasing support
///    - Remove infrequent children
///
/// @param C                 current node
/// @param iTAIL             index into tail of current node
/// @param useComp           whether compressed bitmaps should be used
/// @param NoChild           whether C has any frequent children
/// @param AllFreq           whether all children are frequent
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::ReorderTail(
    TreeNode *C,
    int &iTAIL,
    bool useComp,
    bool &NoChild,
    bool &AllFreq) {

    int tailIndex = 0;
    int lol = 0;
    // for each tail element
    for (lol = iTAIL; lol < C->tEnd; lol++) {
        Bitmap *trans = TransBuffy[C->Depth];
        int theCount = 0;

        // Compress the bitmaps
        if (useComp && (F1[gTail[lol].Item]->compID != C->compID)) {
            F1[gTail[lol].Item]->Trans->BuildRelComp(
                *TransBuffy[projectDepth]);
            F1[gTail[lol].Item]->compID = C->compID;
            CountRebuilds++;
        }

        // Use the compressed bitmaps
        if (useComp) {
            // AND the compressed bitmaps and count the result
            trans->AndCompOnly(
                *C->Trans,
                *F1[gTail[lol].Item]->Trans,
                CountSmallAnds);
            theCount = trans->SmallCount(CountCounts);

            // use the full bitmaps
        } else {
            // AND & count the bitmaps
            trans->AndOnly(*C->Trans, *F1[gTail[lol].Item]->Trans, CountAnds);
            theCount = trans->Count(CountCounts);
        }

        // If the results is frequent
        if (theCount >= MS) {
            // if PEP pruning holds
            if (PEPrune && (trans->_count == C->Trans->_count)) {
                // Move tail element from tail to head
                C->Name->Or(*C->Name, *F1[gTail[lol].Item]->Name);
                CountPEPrunes++;

                // add tail element to reordered tail
            } else {
                NoChild = false;

                // create new tail element
                TailBuffy[tailIndex].Count = theCount;
                TailBuffy[tailIndex].Item = gTail[lol].Item;
                tailIndex++;
            }
        } else
            AllFreq = false;
    }

    sort(TailBuffy, TailBuffy + tailIndex);

    // Set the begin and end values of the new tail
    iTAIL = C->tEnd;
    C->tEnd = iTAIL + tailIndex;
    C->tBegin = iTAIL;
    int r = 0;

    // Copy new tail into next slots
    for (lol = iTAIL; lol < C->tEnd; lol++) {
        gTail[lol] = TailBuffy[r];
        r++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Simply copy over the tail without expanding any of the children
///    for pure DFS (no expansion of all children)
///
/// @param C                 current node
/// @param iTAIL             index into tail of current node
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::NoorderTail(TreeNode *C, int &iTAIL) {

    // set begin and end tail pointers
    iTAIL = C->tEnd;
    C->tEnd = C->tEnd - C->tBegin + C->tEnd;
    C->tBegin = iTAIL;
    int r = 0;

    // copy over old tail to new tail
    for (int lol = iTAIL; lol < C->tEnd; lol++) {
        gTail[lol] = gTail[lol - C->tEnd + C->tBegin];
        r++;
    }
}


/////////////////////////////////////////////////////////////////////
/// The main MAFIA algorithm function
///
/// @param C                 the current node
/// @param HUT               whether this is a HUT check (left most branch)
/// @param FHUT              [output] whether the HUT is frequent
/// @param useComp           if compression has been switched on
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::MAFIA(TreeNode *C, bool HUT, bool &FHUT, bool useComp) {
    CountNodes++;
    int iTAIL = C->tBegin;   // index into the tail
    bool NoChild = true;     // whether the node has any frequent children
    bool AllFreq = true;     // whether all the children are frequent
    FHUT = false;            // whether this is a FHUT
    int beforeCountNodes = CountNodes;
    int frequentTailSize = C->tEnd - iTAIL;

    if (iTAIL < C->tEnd) {

        if (C != Root) {
            if (Reorder) {
                ReorderTail(C, iTAIL, useComp, NoChild, AllFreq);
            } else {
                NoorderTail(C, iTAIL);
            }
        }

        frequentTailSize = C->tEnd - iTAIL;

        int estimateTail = (int)(frequentTailSize / (double)EstimateDiv);
        if (estimateTail > 0 && C->Trans->_count != TransCount) {
            double estimateSubTree = EstimateBuffy[estimateTail].Sum / (double)EstimateBuffy[estimateTail].Count;
            double support = C->Trans->_count / (double) TransCount;
            double factor = 11.597 - 29.914 * (support - .52392) * (support - .52392);
            double cost = abs(factor * frequentTailSize / (1 - 1.2 * support));
            //double cost = 5 * frequentTailSize / (1 - support);

            // check if relative comp should be performed
            if ((!useComp) && (estimateSubTree > cost)) {

                // build the relative bitmap for source [node bitmap] (all 1's)
                C->Trans->BuildSource();

                // remember the depth of the FULL bitmap your projecting from
                projectDepth = C->Depth - 1;

                // increment the ID
                C->compID = MAX_compID;
                MAX_compID++;
                useComp = true;
            }
        }

        // Candidate generation - extend the Head with the tail elements
        // We start from the end of the tail and move backwards
        // Therefore the tail is iterated through in increasing support,
        // but is stored in decreasing support.
        while (iTAIL < C->tEnd) {
            // form a one-extension
            Bitmap *trans = TransBuffy[C->Depth];
            BaseBitmap *name = NameBuffy[C->Depth];
            TreeNode *newNode = NodeBuffy[C->Depth];

            // create name for the new node
            name->Or(*C->Name, *F1[gTail[iTAIL].Item]->Name);

            // compress the bitmaps if warranted
            if (useComp && (F1[gTail[iTAIL].Item]->compID != C->compID)) {
                // build the relative for this node
                F1[gTail[iTAIL].Item]->Trans->BuildRelComp(
                    *TransBuffy[projectDepth]);
                F1[gTail[iTAIL].Item]->compID = C->compID;
                CountRebuilds++;
            }

            int theCount = 0;

            // use the compressed bitmaps for ANDing and counting
            if (useComp) {
                // AND and count small bitmaps
                trans->AndCompOnly(
                    *C->Trans,
                    *F1[gTail[iTAIL].Item]->Trans,
                    CountSmallAnds);

                if (Reorder)
                    trans->_count = gTail[iTAIL].Count;
                else
                    theCount = trans->SmallCount(CountCounts);
            } else {
                // AND and count the full bitmaps
                trans->AndOnly(
                    *C->Trans,
                    *F1[gTail[iTAIL].Item]->Trans,
                    CountAnds);

                if (Reorder)
                    trans->_count = gTail[iTAIL].Count;
                else
                    theCount = trans->Count(CountCounts);
            }
            if (!Reorder && PEPrune && (theCount == C->Trans->_count)) {
                CountPEPrunes++;
                C->Name->Or(*C->Name, *F1[gTail[iTAIL].Item]->Name);
                iTAIL++;
                continue;
            }

            // Determine whether this candidate will be a HUT
            // Conceptually the leftmost branch of the tree is a HUT check
            if ((iTAIL != C->tBegin) && (C != Root))
                HUT = 0;
            else
                HUT = 1;

            if (!AllFreq)
                HUT = 0;

            if (Reorder || (theCount >= MS)) {
                // form the 1-extension node
                newNode->setTreeNode(name,
                                     trans,
                                     C->Depth + 1,
                                     C->compID,
                                     F1[gTail[iTAIL].Item]->Prefix,
                                     iTAIL + 1,
                                     C->tEnd);

                // setup the LMFI for the next level; it contains all
                // itemsets in LMFI for this level that also include the
                // one we're extending the node with.  We do sort of a
                // quicksort thing to move the relevant itemsets for the
                // next node to the end of the portion of the MFI relevant
                // to this node

                newNode->rEnd = C->rEnd;
                newNode->rBegin = SortLMFI(C->rBegin, C->rEnd, newNode->Prefix);

                // Check for HUT in MFI for remaining tail
                if (HUTMFI && newNode->tBegin != newNode->tEnd && !HUT) {
                    CountHUTMFI++;

                    if (CheckHUTMFI(newNode, newNode->tBegin)) {
                        // stop generation of extensions
                        CountHUTMFISuccess++;
                        AllFreq = false;
                        break;
                    }
                }

                NoChild = false;

                // recurse down the tree
                MAFIA(newNode, HUT, FHUT, useComp);

                // Add those discovered from lower levels to the current LMFI
                // LMFI_l = LMFI_l \union LMFI_{l+1}
                // all we need to do is to update the end pointer
                C->rEnd = newNode->rEnd;
            } else
                AllFreq = false;

            // if this was a successful HUT check
            if (FHUT) {
                // keep going up the tree
                if (HUT) {
                    return ;

                    // reached start of HUT, so stop generation of subtree
                    // rooted at this node
                } else {
                    FHUT = false;
                    break;
                }
            }

            // Move on the next tail element
            iTAIL++;
        }
    }

    // if this is a FHUT
    if (GoFHUT && HUT && AllFreq) {
        FHUT = true;
        CountFHUT++;
    }

    // if this node is childless and not in MFI
    if (MethodIsFI)
        AddToFI(C);
    else if (MethodIsFCI && C != Root)
        AddToFCI(C);
    else if (NoChild && !LMFISuperSet(C)) {
        AddToMFI(C);
    }

    int subtreeSize = CountNodes - beforeCountNodes + 1;
    int estimateTail = (int)(frequentTailSize / (double)EstimateDiv);
    if (estimateTail > 0 && C->Trans->_count != TransCount) {
        EstimateBuffy[estimateTail].Count++;
        EstimateBuffy[estimateTail].Sum += subtreeSize;
    }
}


/////////////////////////////////////////////////////////////////////
/// Merge repeated itemsets into one combined itemset
///    - e.g. if (transaction set of item 4) AND
///    (transaction set item 5) = (transaction set item 5),
///    then item 5 is a duplicate of item 4
///    due to increasing support
/////////////////////////////////////////////////////////////////////
void MAFIA::Mafia::MergeRepeatedItemsets() {
    NodeList::iterator bali = F1.begin();
    Bitmap out(*(*bali)->Trans);
    int blah = 0;

    // for each frequent 1-itemset
    while (bali != F1.end()) {
        out.FillOnes();
        NodeList::iterator noli = bali;
        noli++;

        // search for a copy of the itemset's transaction set
        while (noli != F1.end()) {
            // stop when count is no longer the same
            if ((*bali)->Trans->_count != (*noli)->Trans->_count)
                break;
            else {
                // AND itemsets with the same count
                out.AndOnly(*(*bali)->Trans, *(*noli)->Trans, CountAnds);
                out.Count(blah);

                // check for a duplicate
                if (out._count == (*noli)->Trans->_count) {
                    (*bali)->Name->Or(*(*bali)->Name, *(*noli)->Name);
                    F1.erase(noli);
                } else
                    noli++;
            }
        }
        bali++;
    }
}
/** @} */
