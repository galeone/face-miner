#ifndef MAFIA_H
#define MAFIA_H

#include <cstring>
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include <list>
#include <vector>
#include <algorithm>
#include <string>
#include <assert.h>
#include <cmath>
#include <map>
#include "Bitmap.h"
#include "TreeNode.h"
#include "Transaction.h"
#include "ItemsetOutput.h"

namespace MAFIA {

    class Mafia{

    private:

    public:

        typedef vector<TreeNode *> NodeList;
        typedef vector<TreeNode *> BranchList;
        typedef vector<Bitmap *> BitmapList;
        typedef vector<BaseBitmap *> BaseBitmapList;
        typedef vector<int> ItemSet;
        typedef map<long, ItemSet *> HashTable;

        /// Simple class for storing subtree size estimates
        class SubtreeEstimate {
        public:
            int Count;              ///< Count of actual subtrees counted
            int Sum;                ///< Sum of subtree sizes
            SubtreeEstimate () {
                Count = 0;
                Sum = 0;
            }
        };

        /// Simple class for storing tail elements of each node of the tree
        class TailElement {
        public:
            int Count;              ///< Support of the 1-extension
            int Item;               ///< Item-id for this1 -extension

            TailElement () {
                Count = 0;
                Item = 0;
            }

            bool operator < (const TailElement& rhs) const {
                return this->Count < rhs.Count;
            };
        };


        void AddToF1(TreeNode *newNode);
        void F1UsingProb(double P);
        void F1FromFile(char *filename, bool isAsciiFile);
        void PrintMFI();
        bool LMFISuperSet(TreeNode* location);
        void AddToFI(TreeNode *C);
        void AddToMFI(TreeNode *C);
        void AddToFCI(TreeNode *C);
        int SortLMFI(int rBegin, int rEnd, int sortBy);
        bool CheckHUTMFI(TreeNode *C, int iTAIL);
        void ReorderTail(TreeNode *C,
                             int &iTAIL,
                             bool useComp,
                             bool &NoChild,
                             bool &AllFreq);
        void NoorderTail(TreeNode *C, int &iTAIL);
        void MAFIA(TreeNode *C, bool HUT, bool &FHUT, bool useComp);
        void MergeRepeatedItemsets();

    };

}

#endif // MAFIA_H

