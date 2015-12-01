#include "facepatternminer.h"
FacePatternMiner::FacePatternMiner(QString dataset, QString mimeFilter) {
    _mimeFilter = mimeFilter;
    _it = new QDirIterator(dataset);
    _edgeDir = new QDir("edge");
    if(!_edgeDir->exists()) {
        QDir().mkdir(_edgeDir->path());
    }
}

inline bool FacePatternMiner::_validMime(QString fileName) {
    QMimeDatabase mimeDB;
    return  mimeDB.mimeTypeForFile(fileName).inherits(_mimeFilter);
}

void FacePatternMiner::_appendToTestSet(const cv::Mat &) {

}

// Costruire il training database (proprio un file che contiene le edge image estratte)
/*Datasets can be in ASCII or binary format. For ASCII files, the file format must be:
[item_id_1] [item_id_2] ... [item_id_n]
Items do not have to be sorted within each transaction. Items are separated by spaces and each transaction should end with a newline, e.g.
1 4 2
2 8 9 4
2 5
*/
// Dato che usa gli spazi come separatori di item e i \n come separatore di transaction, creiamo il dataset in questo formato
// (x,y) dove x,y sono le coordiante del px bianco nella i-esima immagine
void FacePatternMiner::_preprocess() {
    while(_it->hasNext()) {
        auto fileName = _it->next();
        if(!_validMime(fileName)) {
            continue;
        }

        auto image = cv::imread(fileName.toStdString());
        emit preprocessing(image);
        // lets user the histogram equalization method in order to
        // equalize the distribution of greys in the original image
        // Thus we stretch the historgram trying to make it plan

        // first, convert the image to grayscale if is not in grayscale already
        if(image.channels() > 1) {
            cv::cvtColor(image, image, CV_BGR2GRAY);
        }

        // second, equalize it
        cv::Mat equalizedImage;
        cv::equalizeHist(image, equalizedImage);

        // TODO: emettere un altro segnale e mostrare immagine affiancata frutto del preproressing
        emit preprocessing(equalizedImage);

        // now we can use the sobel operator to extract the edges of the equalized image
        // sobel operator calculate an approximation of the partial derivates, in order to find out
        // the light changing in an pixel neighborhood

        cv::Mat grad_x, grad_y;
        // from the equalized image, save into grad_x the derivate along the x axes (order 1) (order 0 to y axes)
        // uses depth of 16 bit signed, to avoid overflow (the derivate can be less then zero)
        cv::Sobel(equalizedImage, grad_x, CV_16S, 1, 0);
        // the same of above, but along the y axes
        cv::Sobel(equalizedImage, grad_y, CV_16S, 0, 1);

        // converting from 16S to 8U (255 shades of gray LOL)
        cv::Mat grad_x_abs, grad_y_abs;
        cv::convertScaleAbs(grad_x, grad_x_abs);
        cv::convertScaleAbs(grad_y, grad_y_abs);

        // try to approximate the gradient, using a weighted sum of the calculated partial derivates
        // The approxiamation is G = |G_x| + |G_y|
        cv::Mat grad;
        cv::addWeighted(grad_x_abs, 1, grad_y_abs, 1, 0, grad);

        emit preprocessing(grad);

        // Now we apply a segmentation algorithm, in order to remove noise from the grayscale equalized edge detected image
        cv::Scalar mu, sigma;

        cv::meanStdDev(grad,mu,sigma);

        // Thresholding
        const double c = 1;
        // Scalar is a vector of quartets, we're working on grayscale thus we extract only the first channel
        double threshold = mu[0] + c * sigma[0];
        cv::Mat thresRes;
        cv::threshold(grad,thresRes, threshold, 255, CV_THRESH_BINARY);

        emit preprocessing(thresRes);

        // last step of preprocessing, dilatation
        int structuringMatrix[3][3] = {
            {0,1,0},
            {1,1,1},
            {0,1,0}
        };
        cv::Mat structuringElement(3,3, CV_8UC1, &structuringMatrix);
        cv::Mat dilatationRes;
        cv::dilate(thresRes,dilatationRes,structuringElement);

        emit preprocessing(dilatationRes);
#ifdef DEBUG
        std::string edgeFile = _edgeDir->absolutePath().append(QString("/")).append(fileName.split(QString("/")).last()).toStdString();
        cv::imwrite(edgeFile,dilatationRes);
#endif
        // Appending transaction (the image), to transaction database
        _appendToTestSet(dilatationRes);
    }

}

void FacePatternMiner::_mineMFI() {/*
    // Check parameters
    MAFIA::Mafia mafia;
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " [-mfi/-fci/-fi] [min sup (percent)] " << endl;
        cerr << "\t[-ascii/-binary] [input filename] " << endl;
        cerr << "\t[output filename (optional)]" << endl;
        cerr << "Ex: " << argv[0] << " -mfi .5 -ascii connect4.ascii mfi.txt" << endl;
        cerr << "Ex: " << argv[0] << " -mfi .3 -binary chess.binary" << endl;
        exit(0);
    }

    // time hook
    time(&total_start);

    // get the algorithm type
    method = argv[1];

    // Minimum support as a fraction
    MSF = atof(argv[2]);

    if (strcmp(argv[3], "-ascii") == 0)
        F1FromFile(argv[4], true);
    else if (strcmp(argv[3], "-binary") == 0)
        mafia.F1FromFile(argv[4], false);
    else {
        cerr << "File format must be -ascii or -binary" << endl;
        exit(1);
    }

    if (argc == 6) {
        outputMFI = true;
        outFilename = argv[5];
    }

    // Create a null node to begin the DFS tree
    NullTrans = new Bitmap(TransCount);
    NullTrans->FillOnes();
    NullTrans->_count = TransCount;
    ItemSet NullList;

    MFIBySizes = new int[MAX_ITEMSET_SIZE];
    for (int h = 0; h < MAX_ITEMSET_SIZE; h++) {
        MFIBySizes[h] = 0;
    }

    // Set size of F1
    FullF1size = F1size = F1.size();

    // if F1 is not empty
    if (FullF1size != 0) {

        BaseBitmap *NullName = new BaseBitmap(FullF1size);
        MFI.reserve(100000);

        int p = 0;
        ItemMap = new int[FullF1size];
        ItemsetBuffy = new int[FullF1size];

        // Rename items in F1
        for (NodeList::iterator nli = F1.begin(); nli != F1.end(); nli++) {
            // store old itemid
            ItemMap[p] = (*nli)->Prefix;

            // assign new itemid
            (*nli)->Prefix = p;

            // assign name bitmaps
            (*nli)->Name = new BaseBitmap(FullF1size);
            (*nli)->Name->FillEmptyPosition(p);
            (*nli)->Name->_count = 1;
            p++;
        }

        // don't merge equivalent items for FI output
        if (method.compare("-fi") != 0) {
           MergeRepeatedItemsets();
        }

        F1size = F1.size();

        // Create global tail
        maxtail = F1size * (F1size + 1) / 2;
        gTail = new TailElement[maxtail];

        // Create buffer for sorting
        TailBuffy = new TailElement[F1size];

        // Create buffer for estimating size of each subtree
        EstimateSize = (int)ceil(F1size / (double)EstimateDiv);
        EstimateBuffy = new SubtreeEstimate[EstimateSize];
        for (int estimateIndex = 0; estimateIndex < EstimateSize; estimateIndex++) {
            EstimateBuffy[estimateIndex].Count = 1;
            EstimateBuffy[estimateIndex].Sum = estimateIndex * EstimateDiv * estimateIndex * EstimateDiv / 2;
        }

        // Initialize global tail
        int uu;
        for (uu = 0; uu < maxtail; uu++) {
            gTail[uu].Item = -1;
            gTail[uu].Count = 0;
        }

        // Fill global tail
        for (uu = 0; uu < F1size; uu++) {
            gTail[uu].Item = uu;
            gTail[uu].Count = F1[uu]->Trans->_count;

            // assign tail index
            F1[uu]->tBegin = uu + 1;
            if (uu == F1size - 1)
                F1[uu]->tBegin = -1;

            TempName = new BaseBitmap(FullF1size);

            // add a buffer element for each item in F1
            BaseBitmap *name = new BaseBitmap(FullF1size);
            NameBuffy.push_back(name);
            Bitmap *buff = new Bitmap(TransCount);
            TransBuffy.push_back(buff);
            TreeNode *newNode = new TreeNode();
            NodeBuffy.push_back(newNode);
        }

        srand(666);
        bool FHUT;

        // start algorithm timer
        clock_t start, finish;
        double duration = -1;
        start = clock();

        time(&algorithm_start);

        // create root node and its associated tail
        Root = new TreeNode(NullName, NullTrans, 0, 0, -1, 0, F1size);

        //Nothing is in MFI, so nothing is relevant
        Root->rBegin = 0;
        Root->rEnd = 0;

        // run the appropriate algorithm
        if (method.compare("-fci") == 0) {
            //cout << "running closure (FCI) algorithm..." << endl;

            GoFHUT = false;      // FHUT flag
            HUTMFI = false;      // HUTMFI flag
            PEPrune = true;      // PEPrune flag
            Reorder = true;      // Reorder flag
            MethodIsFCI = true;

            MAFIA(Root, false, FHUT, false);
        } else if (method.compare("-mfi") == 0) {
            //cout << "running MFI algorithm..." << endl;
            GoFHUT = true;      // FHUT flag
            HUTMFI = true;      // HUTMFI flag
            PEPrune = true;     // PEPrune flag
            Reorder = true;     // Reorder flag

            MAFIA(Root, true, FHUT, false);
        } else if (method.compare("-fi") == 0) {
            if (outputMFI) {
                // open output file
                outFile = new ItemsetOutput(outFilename);
                if (!outFile->isOpen()) {
                    cerr << "Output file not open!" << endl;
                    exit(1);
                }
            }

            //cout << "running FI algorithm..." << endl;
            GoFHUT = false;      // FHUT flag
            HUTMFI = false;      // HUTMFI flag
            PEPrune = false;     // PEPrune flag
            Reorder = false;     // Reorder flag
            MethodIsFI = true;

            MAFIA(Root, false, FHUT, false);

            if (outputMFI) {
                delete outFile;
            }
        } else {
            cerr << "Invalid algorithm option!" << endl;
            exit(0);
        }

        finish = clock();
        duration = (finish - start) / (double)CLOCKS_PER_SEC;

        //printf( "Algorithm CPU time: %.3f seconds.\n", duration );

        time(&algorithm_finish);
        algorithm_time = difftime(algorithm_finish, algorithm_start);
        printf("Algorithm time:        %.2f seconds.\n", algorithm_time);
    } else {
        MFIBySizes[0]++;
    }


    // Print out MFI length distribution
    //for (int sizeIndex = 0; sizeIndex <= maxItemsetSize; sizeIndex++) {
    //    cout << sizeIndex << " " << MFIBySizes[sizeIndex] << endl;
    //}


    if (outputMFI && !MethodIsFI) {
        // PrintMFI to a file
        //cout << "Printing out the mfi..." << endl;
        time(&print_start);

        PrintMFI();

        time(&print_finish);
        print_time = difftime(print_finish, print_start);
        printf("Printing output time:  %.2f seconds.\n", print_time);
    }

    time(&total_finish);
    total_time = difftime(total_finish, total_start);
    printf("Total time:            %.2f seconds.\n\n", total_time);

    // output stat data
    cout << "MinSup:      " << MS << endl;
    cout << "TransCount:  " << TransCount << endl;
    cout << "F1size:      " << FullF1size << endl;
    if (!MethodIsFI)
        cout << "MFISize:     " << MFI.size() << endl;
    else
        cout << "MFISize:     " << MFISize << endl;

    cout << "MFIAvg:      " << MFIDepth / (double) MFISize << endl << endl;
    */
}

// slot
void FacePatternMiner::start() {
    _preprocess();
    emit proprocessing_terminated();
    _mineMFI();
    emit mining_terminated();
}
